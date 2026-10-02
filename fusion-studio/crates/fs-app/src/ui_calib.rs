
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fs_calib::board::BoardKind;
use fs_calib::geom::validate_shared_target;
use fs_calib::GeomMode;

use crate::calib_worker::{CalibUiState, CamPanel, ResetKind};


const OVERLAY_FADE: Duration = Duration::from_secs(1);


pub fn active_calib_section(
    ui: &mut egui::Ui,
    status: &str,
    load_path: &mut String,
    can_validate: bool,
    calib: Option<&Arc<Mutex<CalibUiState>>>,
    msg: Option<&Result<String, String>>,
) -> bool {
    ui.label(status);
    ui.text_edit_singleline(load_path);
    let mut load_clicked = false;
    ui.horizontal(|ui| {
        if ui
            .add_enabled(can_validate, egui::Button::new("加载"))
            .on_disabled_hover_text("等第一帧到来后才能校验尺寸")
            .clicked()
        {
            load_clicked = true;
        }
        if let Some(state) = calib {
            let mut c = state.lock().unwrap();
            let can_apply = c.stereo.is_some() && !c.frozen;
            let hover = if c.frozen { "失步冻结中,worker 不会处理" } else { "还没有完整的双目解" };
            if ui
                .add_enabled(can_apply, egui::Button::new("应用此会话结果"))
                .on_disabled_hover_text(hover)
                .clicked()
            {
                c.apply_request = true;
            }
        }
    });
    match msg {
        Some(Ok(m)) => {
            ui.colored_label(egui::Color32::from_rgb(60, 200, 90), m);
        }
        Some(Err(e)) => {
            ui.colored_label(egui::Color32::RED, e);
        }
        None => {}
    }
    ui.separator();
    load_clicked
}


pub fn geometry_section(
    ui: &mut egui::Ui,
    current: GeomMode,
    target: &mut (u32, u32),
    gate: Option<&'static str>,
    msg: Option<&Result<String, String>>,
) -> Option<GeomMode> {
    ui.strong("几何处理");
    let mut request = None;
    if let Some(reason) = gate {
        ui.weak(reason);
    }
    ui.add_enabled_ui(gate.is_none(), |ui| {
        ui.horizontal(|ui| {
            let (off, und, shk) = (
                matches!(current, GeomMode::Off),
                matches!(current, GeomMode::Undistort),
                matches!(current, GeomMode::SharedK { .. }),
            );
            if ui.radio(off, "关闭").clicked() && !off {
                request = Some(GeomMode::Off);
            }
            if ui.radio(und, "去畸变 Undistort").clicked() && !und {
                request = Some(GeomMode::Undistort);
            }
            if ui.radio(shk, "共享内参 Shared-K").clicked() && !shk {
                request = Some(GeomMode::SharedK { target: *target });
            }
        });
        ui.horizontal(|ui| {
            ui.label("目标 W/H");
            let rw = ui.add(egui::DragValue::new(&mut target.0).range(16..=4094).speed(2));
            let rh = ui.add(egui::DragValue::new(&mut target.1).range(16..=4094).speed(2));
            if let GeomMode::SharedK { target: applied } = current {
                let done = rw.drag_stopped() || rw.lost_focus() || rh.drag_stopped() || rh.lost_focus();
                if done && *target != applied && validate_shared_target(*target).is_ok() {
                    request = Some(GeomMode::SharedK { target: *target });
                }
            }
        });
    });
    if gate.is_none() {
        if let Err(e) = validate_shared_target(*target) {
            ui.colored_label(egui::Color32::RED, format!("目标尺寸无效:{e}"));
        }
    }
    match msg {
        Some(Ok(m)) => {
            ui.colored_label(egui::Color32::from_rgb(60, 200, 90), m);
        }
        Some(Err(e)) => {
            ui.colored_label(egui::Color32::RED, e);
        }
        None => {}
    }
    ui.separator();
    request
}

pub fn calibrate_panel(
    ui: &mut egui::Ui,
    state: &Arc<Mutex<CalibUiState>>,
    export_path: &mut String,
    show_coverage: &mut bool,
    geometry_on: bool,
) -> bool {
    let mut start_pending_geometry_off = false;
    let mut c = state.lock().unwrap();

    if let Some(fault) = c.geometry_fault {
        ui.colored_label(egui::Color32::RED, fault);
    }

    let board_editable = !c.session_exists;

    ui.label("board");
    ui.add_enabled_ui(board_editable, |ui| {
        ui.horizontal(|ui| {
            let is_charuco = matches!(c.board.kind, BoardKind::Charuco { .. });
            egui::ComboBox::from_id_salt("calib_board_kind")
                .selected_text(if is_charuco { "Charuco" } else { "Checkerboard" })
                .show_ui(ui, |ui| {
                    if ui.selectable_label(!is_charuco, "Checkerboard").clicked() && is_charuco {
                        c.board.kind = BoardKind::Checkerboard;
                    }
                    if ui.selectable_label(is_charuco, "Charuco").clicked() && !is_charuco {
                        c.board.kind = BoardKind::Charuco { marker_mm: 20.0, dict: "DICT_4X4_50".to_string() };
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("rows");
            ui.add(egui::DragValue::new(&mut c.board.inner_rows).range(2..=20));
            ui.label("cols");
            ui.add(egui::DragValue::new(&mut c.board.inner_cols).range(2..=20));
        });
        ui.horizontal(|ui| {
            ui.label("rect w/h mm");
            ui.add(egui::DragValue::new(&mut c.board.rect_w_mm).range(1.0..=1000.0).speed(0.5));
            ui.add(egui::DragValue::new(&mut c.board.rect_h_mm).range(1.0..=1000.0).speed(0.5));
        });
        if let BoardKind::Charuco { marker_mm, dict } = &mut c.board.kind {
            ui.horizontal(|ui| {
                ui.label("marker mm");
                ui.add(egui::DragValue::new(marker_mm).range(1.0..=1000.0).speed(0.5));
            });
            ui.horizontal(|ui| {
                ui.label("dict");
                ui.text_edit_singleline(dict);
            });
        }
    });
    if !board_editable {
        ui.colored_label(ui.visuals().weak_text_color(), "Reset all stops calibration and frees the board (edits apply on next Start)");
    }

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button(if c.enabled { "Pause" } else { "Start" }).clicked() {
            if !c.enabled && geometry_on {
                start_pending_geometry_off = true;
            } else {
                c.enabled = !c.enabled;
            }
        }
        ui.label(if c.enabled { "running" } else { "paused" });
        if c.calib_in_flight {
            ui.colored_label(ui.visuals().weak_text_color(), "calibrating…");
        }
    });
    ui.label(format!("synced {} | recon pairing mismatches {}", c.synced_seen, c.recon_pair_mismatches));

    cam_section(ui, "FLIR (cam0)", &c.cam0);
    cam_section(ui, "EVK4 (cam1)", &c.cam1);

    ui.separator();
    ui.horizontal(|ui| {
        ui.strong("Stereo");
        converged_dot(ui, c.stereo.is_some());
    });
    match c.stereo {
        Some((rms, angle_deg, t)) => {
            let t_norm = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            ui.label(format!("rms={rms:.3}px angle={angle_deg:.2}\u{b0} |T|={t_norm:.1}mm"));
        }
        None => {
            ui.label("not calibrated");
        }
    }

    ui.separator();
    ui.checkbox(show_coverage, "show coverage heatmap on previews");

    ui.separator();
    ui.horizontal(|ui| {
        if ui.add_enabled(c.stereo.is_some(), egui::Button::new("Reset stereo")).clicked() {
            c.reset_request = Some(ResetKind::Stereo);
        }
        if ui.add_enabled(c.cam0.pool_len > 0, egui::Button::new("Reset cam0")).clicked() {
            c.reset_request = Some(ResetKind::Camera(0));
        }
        if ui.add_enabled(c.cam1.pool_len > 0, egui::Button::new("Reset cam1")).clicked() {
            c.reset_request = Some(ResetKind::Camera(1));
        }
    });
    if ui.add_enabled(c.session_exists, egui::Button::new("Reset all")).clicked() {
        c.reset_request = Some(ResetKind::All);
    }

    ui.separator();
    ui.label("export");
    ui.text_edit_singleline(export_path);
    if ui.button("Export").clicked() {
        c.export_request = Some(export_path.clone());
    }
    match &c.export_result {
        Some(Ok(msg)) => {
            ui.colored_label(egui::Color32::from_rgb(60, 200, 90), msg);
        }
        Some(Err(e)) => {
            ui.colored_label(egui::Color32::RED, e);
        }
        None => {}
    }
    start_pending_geometry_off
}

fn cam_section(ui: &mut egui::Ui, label: &str, cam: &CamPanel) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.strong(label);
        converged_dot(ui, cam.converged);
    });
    ui.add(egui::ProgressBar::new(cam.pool_len as f32 / cam.cap.max(1) as f32).text(format!("{}/{}", cam.pool_len, cam.cap)));
    let rms = cam.rms.map(|v| format!("{v:.3}px")).unwrap_or_else(|| "-".to_string());
    ui.label(format!("rms: {rms}"));
    ui.label(format!("fx={:.1} fy={:.1} cx={:.1} cy={:.1}", cam.fx, cam.fy, cam.cx, cam.cy));

    let fx: Vec<f64> = cam.history.iter().map(|h| h[0]).collect();
    let fy: Vec<f64> = cam.history.iter().map(|h| h[1]).collect();
    let rms_hist: Vec<f64> = cam.history.iter().map(|h| h[4]).collect();
    history_curve(ui, 36.0, &[("fx", egui::Color32::from_rgb(90, 200, 120), &fx), ("fy", egui::Color32::from_rgb(110, 160, 250), &fy)]);
    history_curve(ui, 26.0, &[("rms", egui::Color32::from_rgb(235, 130, 80), &rms_hist)]);
}

fn converged_dot(ui: &mut egui::Ui, converged: bool) {
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    let color = if converged { egui::Color32::from_rgb(60, 200, 90) } else { egui::Color32::GRAY };
    ui.painter().circle_filled(rect.center(), 4.0, color);
}


fn history_curve(ui: &mut egui::Ui, height: f32, series: &[(&str, egui::Color32, &Vec<f64>)]) {
    let width = ui.available_width().min(230.0);
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    for (_, color, vals) in series {
        if vals.len() < 2 {
            continue;
        }
        let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let span = (hi - lo).max(1e-9);
        let pts: Vec<egui::Pos2> = vals
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = rect.left() + (i as f32 / (vals.len() - 1) as f32) * rect.width();
                let y = rect.bottom() - ((v - lo) / span) as f32 * rect.height();
                egui::pos2(x, y)
            })
            .collect();
        painter.line(pts, egui::Stroke::new(1.5, *color));
    }
    ui.horizontal(|ui| {
        for (name, color, _) in series {
            ui.colored_label(*color, *name);
        }
    });
}


const ACCEPTED_COLOR: (u8, u8, u8) = (80, 220, 120);
const DETECTED_ONLY_COLOR: (u8, u8, u8) = (170, 170, 170);


pub fn draw_pane_overlay(
    ui: &egui::Ui,
    rect: egui::Rect,
    tex_size: egui::Vec2,
    corners: Option<&[(f32, f32)]>,
    corners_age: Option<Duration>,
    accepted: bool,
    coverage: [[bool; 6]; 6],
    show_heatmap: bool,
) {
    if tex_size.x <= 0.0 || tex_size.y <= 0.0 {
        return;
    }
    let painter = ui.painter_at(rect);

    if show_heatmap {
        for (row, cells) in coverage.iter().enumerate() {
            for (col, &covered) in cells.iter().enumerate() {
                if covered {
                    continue;
                }
                let cell = egui::Rect::from_min_max(
                    rect.min + egui::vec2(col as f32 / 6.0 * rect.width(), row as f32 / 6.0 * rect.height()),
                    rect.min + egui::vec2((col + 1) as f32 / 6.0 * rect.width(), (row + 1) as f32 / 6.0 * rect.height()),
                );
                painter.rect_filled(cell, 0.0, egui::Color32::from_rgba_unmultiplied(220, 40, 40, 55));
            }
        }
    }

    if let (Some(pts), Some(age)) = (corners, corners_age) {
        if age < OVERLAY_FADE {
            let alpha = (1.0 - age.as_secs_f32() / OVERLAY_FADE.as_secs_f32()).clamp(0.0, 1.0);
            let (r, g, b) = if accepted { ACCEPTED_COLOR } else { DETECTED_ONLY_COLOR };
            let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(r, g, b, (alpha * 255.0) as u8));
            let scale = egui::vec2(rect.width() / tex_size.x, rect.height() / tex_size.y);
            for &(x, y) in pts {
                let p = rect.min + egui::vec2(x * scale.x, y * scale.y);
                painter.circle_stroke(p, 4.0, stroke);
            }
        }
    }
}
