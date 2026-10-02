use crate::settings::AutoMode;

#[derive(Debug, Clone, PartialEq)]
pub enum FlirCmd {
    SetExposureUs(f64),
    SetExposureAuto(AutoMode),
    SetGainDb(f64),
    SetGainAuto(AutoMode),

    SetIspEnable(bool),

    ApplyFps(f64),

    Resync,
}


#[derive(PartialEq, Eq, Clone, Copy)]
enum CoalesceKind {
    Exposure,
    ExposureAuto,
    Gain,
    GainAuto,
    Isp,
}

fn coalesce_kind(c: &FlirCmd) -> Option<CoalesceKind> {
    match c {
        FlirCmd::SetExposureUs(_) => Some(CoalesceKind::Exposure),
        FlirCmd::SetExposureAuto(_) => Some(CoalesceKind::ExposureAuto),
        FlirCmd::SetGainDb(_) => Some(CoalesceKind::Gain),
        FlirCmd::SetGainAuto(_) => Some(CoalesceKind::GainAuto),
        FlirCmd::SetIspEnable(_) => Some(CoalesceKind::Isp),
        FlirCmd::ApplyFps(_) => None,
        FlirCmd::Resync => None,
    }
}


pub fn coalesce_flir_cmds(cmds: Vec<FlirCmd>) -> Vec<FlirCmd> {
    let shadowed: Vec<bool> = (0..cmds.len())
        .map(|i| match coalesce_kind(&cmds[i]) {
            Some(k) => cmds[i + 1..].iter().any(|c| coalesce_kind(c) == Some(k)),
            None => false,
        })
        .collect();
    cmds.into_iter().zip(shadowed).filter_map(|(c, s)| (!s).then_some(c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::AutoMode;

    #[test]
    fn keeps_only_the_last_of_each_live_kind() {
        let got = coalesce_flir_cmds(vec![
            FlirCmd::SetExposureUs(100.0),
            FlirCmd::SetGainDb(1.0),
            FlirCmd::SetExposureUs(200.0),
            FlirCmd::SetExposureUs(300.0),
            FlirCmd::SetGainDb(2.0),
        ]);
        assert_eq!(got, vec![FlirCmd::SetExposureUs(300.0), FlirCmd::SetGainDb(2.0)]);
    }

    #[test]
    fn different_kinds_do_not_shadow_each_other() {
        let got = coalesce_flir_cmds(vec![
            FlirCmd::SetExposureAuto(AutoMode::Continuous),
            FlirCmd::SetGainAuto(AutoMode::Off),
            FlirCmd::SetIspEnable(true),
        ]);
        assert_eq!(got.len(), 3, "四种不同的 kind 互不遮蔽");
    }

    #[test]
    fn apply_fps_is_never_merged_or_reordered() {
        let got = coalesce_flir_cmds(vec![
            FlirCmd::SetExposureUs(100.0),
            FlirCmd::ApplyFps(30.0),
            FlirCmd::SetExposureUs(200.0),
            FlirCmd::ApplyFps(60.0),
        ]);
        assert_eq!(
            got,
            vec![FlirCmd::ApplyFps(30.0), FlirCmd::SetExposureUs(200.0), FlirCmd::ApplyFps(60.0)],
            "只有被后来者覆盖的 SetExposureUs(100.0) 被丢弃,ApplyFps 的相对顺序不变"
        );
    }

    #[test]
    fn empty_and_single_are_unchanged() {
        assert!(coalesce_flir_cmds(vec![]).is_empty());
        assert_eq!(coalesce_flir_cmds(vec![FlirCmd::SetGainDb(3.0)]), vec![FlirCmd::SetGainDb(3.0)]);
    }

    #[test]
    fn resync_is_never_merged_or_reordered() {
        let got = coalesce_flir_cmds(vec![
            FlirCmd::SetExposureUs(100.0),
            FlirCmd::Resync,
            FlirCmd::SetExposureUs(200.0),
            FlirCmd::Resync,
        ]);
        assert_eq!(
            got,
            vec![FlirCmd::Resync, FlirCmd::SetExposureUs(200.0), FlirCmd::Resync],
            "两次 Resync 都必须保留:每一条都是一次完整的重握手"
        );
    }

    #[test]
    fn repeated_isp_toggles_collapse_to_the_last() {
        let got = coalesce_flir_cmds(vec![
            FlirCmd::SetIspEnable(true),
            FlirCmd::SetIspEnable(false),
            FlirCmd::SetIspEnable(true),
        ]);
        assert_eq!(got, vec![FlirCmd::SetIspEnable(true)]);
    }
}
