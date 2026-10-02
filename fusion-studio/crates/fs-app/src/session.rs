
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, TryRecvError};
use fs_core::health::SyncHealth;

use crate::settings::SharedTuning;
use crate::sources::live::LiveHandles;
use crate::wiring::PipelineHandles;
use crate::{ui_recon, ui_settings};

pub const CONNECT_THREAD_DIED: &str = "连接线程异常退出";


pub type ConnectRx<T> = Option<Receiver<Result<T, String>>>;

#[derive(Clone, Debug, PartialEq)]
pub enum ConnectState {
    Disconnected,
    Connecting,
    Connected,
    Failed(String),
}

impl ConnectState {

    pub fn can_start(&self) -> bool {
        matches!(self, ConnectState::Disconnected | ConnectState::Failed(_))
    }

    pub fn button_label(&self) -> &'static str {
        match self {
            ConnectState::Disconnected => "连接相机",
            ConnectState::Connecting => "连接中…",
            ConnectState::Connected => "断开",
            ConnectState::Failed(_) => "重试连接",
        }
    }

    pub fn error(&self) -> Option<&str> {
        match self {
            ConnectState::Failed(e) => Some(e.as_str()),
            _ => None,
        }
    }
}


pub struct AbortGuard {
    shutdown: Arc<AtomicBool>,
    disarmed: bool,
}

impl AbortGuard {

    pub fn arm(shutdown: Arc<AtomicBool>) -> Self {
        AbortGuard { shutdown, disarmed: false }
    }


    pub fn disarm(mut self) {
        self.disarmed = true;
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        if !self.disarmed {
            self.shutdown.store(true, Ordering::Relaxed);
        }
    }
}

pub struct Session {
    pub pipeline: PipelineHandles,
    pub live: Option<LiveHandles>,
    pub tuning: SharedTuning,

    pub health: Arc<Mutex<SyncHealth>>,
    pub recon_panel: ui_recon::ReconPanelState,
    pub settings_panel: ui_settings::SettingsPanelState,

    pub recon_dims: (u32, u32),
    pub shutdown: Arc<AtomicBool>,
}

impl Session {
    pub fn new(
        pipeline: PipelineHandles,
        live: Option<LiveHandles>,
        tuning: SharedTuning,
        settings_ui: ui_settings::SettingsHandles,
        recon_dims: (u32, u32),
        shutdown: Arc<AtomicBool>,
    ) -> Self {
        Session {
            recon_panel: ui_recon::ReconPanelState::new(pipeline.recon.clone(), recon_dims),
            settings_panel: ui_settings::SettingsPanelState::new(settings_ui),
            health: pipeline.health.clone(),
            pipeline,
            live,
            tuning,
            recon_dims,
            shutdown,
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.pipeline.record.shutdown();
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

pub fn poll_connect_result<T>(rx: &mut ConnectRx<T>, state: &mut ConnectState) -> Option<T> {
    let got = rx.as_ref()?.try_recv();
    match got {
        Ok(Ok(v)) => {
            *state = ConnectState::Connected;
            *rx = None;
            Some(v)
        }
        Ok(Err(e)) => {
            *state = ConnectState::Failed(e);
            *rx = None;
            None
        }
        Err(TryRecvError::Empty) => None,
        Err(TryRecvError::Disconnected) => {
            *state = ConnectState::Failed(CONNECT_THREAD_DIED.into());
            *rx = None;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_disconnected_and_can_only_start_once() {
        let mut s = ConnectState::Disconnected;
        assert!(s.can_start(), "未连接时可以发起连接");
        s = ConnectState::Connecting;
        assert!(!s.can_start(), "连接中不能重入");
        s = ConnectState::Connected;
        assert!(!s.can_start(), "已连接不能再连");
        s = ConnectState::Failed("boom".into());
        assert!(s.can_start(), "失败后可以重试");
    }

    #[test]
    fn button_label_tracks_state() {
        assert_eq!(ConnectState::Disconnected.button_label(), "连接相机");
        assert_eq!(ConnectState::Failed("x".into()).button_label(), "重试连接");
        assert_eq!(ConnectState::Connecting.button_label(), "连接中…");
        assert_eq!(ConnectState::Connected.button_label(), "断开");
    }

    #[test]
    fn failure_message_is_preserved_verbatim() {
        let s = ConnectState::Failed("no FLIR camera found (spinError -1004)".into());
        assert_eq!(s.error(), Some("no FLIR camera found (spinError -1004)"));
        assert_eq!(ConnectState::Connecting.error(), None);
    }



    #[test]
    fn poll_ok_yields_session_and_connects() {
        let (tx, rx) = crossbeam_channel::bounded::<Result<(), String>>(1);
        tx.send(Ok(())).unwrap();
        let mut rx = Some(rx);
        let mut st = ConnectState::Connecting;
        assert_eq!(poll_connect_result(&mut rx, &mut st), Some(()));
        assert_eq!(st, ConnectState::Connected);
        assert!(rx.is_none(), "收到结果后通道必须清空");
    }


    #[test]
    fn poll_err_records_reason_verbatim() {
        let (tx, rx) = crossbeam_channel::bounded::<Result<(), String>>(1);
        tx.send(Err("no FLIR camera found".into())).unwrap();
        let mut rx = Some(rx);
        let mut st = ConnectState::Connecting;
        assert_eq!(poll_connect_result(&mut rx, &mut st), None);
        assert_eq!(st, ConnectState::Failed("no FLIR camera found".into()));
        assert!(rx.is_none());
        assert!(st.can_start(), "失败之后必须能重试");
    }

    #[test]
    fn poll_empty_leaves_everything_alone() {
        let (tx, rx) = crossbeam_channel::bounded::<Result<(), String>>(1);
        let mut rx = Some(rx);
        let mut st = ConnectState::Connecting;
        assert_eq!(poll_connect_result(&mut rx, &mut st), None);
        assert_eq!(st, ConnectState::Connecting);
        assert!(rx.is_some(), "连接仍在进行,必须保留通道");
        drop(tx);
    }


    #[test]
    fn poll_disconnected_reports_thread_death() {
        let (tx, rx) = crossbeam_channel::bounded::<Result<(), String>>(1);
        drop(tx);
        let mut rx = Some(rx);
        let mut st = ConnectState::Connecting;
        assert_eq!(poll_connect_result(&mut rx, &mut st), None);
        assert_eq!(st, ConnectState::Failed(CONNECT_THREAD_DIED.into()));
        assert!(rx.is_none());
    }


    #[test]
    fn abort_guard_sets_flag_when_dropped_armed() {
        let flag = Arc::new(AtomicBool::new(false));
        {
            let _g = AbortGuard::arm(flag.clone());
            assert!(!flag.load(Ordering::Relaxed), "arm 本身不置位");
        }
        assert!(flag.load(Ordering::Relaxed), "半路夭折必须置位 shutdown");
    }

    #[test]
    fn abort_guard_leaves_flag_clear_when_disarmed() {
        let flag = Arc::new(AtomicBool::new(false));
        AbortGuard::arm(flag.clone()).disarm();
        assert!(!flag.load(Ordering::Relaxed), "连接成功后标志必须仍然是干净的");
    }
}
