use crate::session::ConnectState;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    Camera,
    Calibrate,
    Record,
}

impl Tab {
    pub const ALL: [Tab; 3] = [Tab::Camera, Tab::Calibrate, Tab::Record];

    pub fn label(self) -> &'static str {
        match self {
            Tab::Camera => "相机",
            Tab::Calibrate => "标定",
            Tab::Record => "录制",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GateReason {
    NotConnected,
    Connecting,
    SyncLost,
}

impl GateReason {

    pub fn message(self) -> &'static str {
        match self {
            GateReason::NotConnected => "未连接相机 —— 请到「相机」tab 连接",
            GateReason::Connecting => "正在连接相机…",
            GateReason::SyncLost => "同步已丢失 —— 请到「相机」tab 重新同步",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PanelGate {
    Open,
    Blocked(GateReason),
}

impl PanelGate {
    pub fn evaluate(tab: Tab, connect: &ConnectState, frozen: bool) -> PanelGate {
        if tab == Tab::Camera {
            return PanelGate::Open;
        }
        match connect {
            ConnectState::Connected if frozen => PanelGate::Blocked(GateReason::SyncLost),
            ConnectState::Connected => PanelGate::Open,
            ConnectState::Connecting => PanelGate::Blocked(GateReason::Connecting),
            _ => PanelGate::Blocked(GateReason::NotConnected),
        }
    }

    pub fn is_open(self) -> bool {
        self == PanelGate::Open
    }

    pub fn reason(self) -> Option<GateReason> {
        match self {
            PanelGate::Blocked(r) => Some(r),
            PanelGate::Open => None,
        }
    }
}


pub fn geom_gate(active_calib: bool, calib_session: bool, recording: bool) -> Option<&'static str> {
    if !active_calib {
        return Some("无当前标定 —— 先在「当前标定」加载或应用一份标定");
    }
    if calib_session {
        return Some("标定会话进行中 —— 标定必须吃原始几何,会话期间强制关闭");
    }
    if recording {
        return Some("录制进行中 —— 不能中途切换几何模式");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::ConnectState;

    #[test]
    fn geom_gate_covers_all_rows() {
        assert_eq!(geom_gate(true, false, false), None);

        for (session, recording) in [(false, false), (true, false), (false, true), (true, true)] {
            let r = geom_gate(false, session, recording).expect("无标定必须拒绝");
            assert!(r.contains("标定"), "{r}");
            assert!(r.contains("加载") || r.contains("应用"), "必须告诉用户怎么解决:{r}");
        }

        for recording in [false, true] {
            let r = geom_gate(true, true, recording).expect("会话期间必须拒绝");
            assert!(r.contains("会话"), "{r}");
        }

        let r = geom_gate(true, false, true).expect("录制中必须拒绝");
        assert!(r.contains("录制"), "{r}");
    }

    #[test]
    fn camera_tab_is_never_blocked() {
        for st in [ConnectState::Disconnected, ConnectState::Connecting, ConnectState::Connected] {
            for frozen in [false, true] {
                assert_eq!(
                    PanelGate::evaluate(Tab::Camera, &st, frozen),
                    PanelGate::Open,
                    "相机 tab 是唯一的自救入口,任何情况下都必须可操作"
                );
            }
        }
    }

    #[test]
    fn calib_and_record_blocked_until_connected() {
        for tab in [Tab::Calibrate, Tab::Record] {
            assert_eq!(
                PanelGate::evaluate(tab, &ConnectState::Disconnected, false),
                PanelGate::Blocked(GateReason::NotConnected)
            );
            assert_eq!(
                PanelGate::evaluate(tab, &ConnectState::Connecting, false),
                PanelGate::Blocked(GateReason::Connecting)
            );
            assert_eq!(PanelGate::evaluate(tab, &ConnectState::Connected, false), PanelGate::Open);
        }
    }

    #[test]
    fn sync_loss_blocks_calib_and_record_even_when_connected() {
        assert_eq!(
            PanelGate::evaluate(Tab::Calibrate, &ConnectState::Connected, true),
            PanelGate::Blocked(GateReason::SyncLost)
        );
        assert_eq!(
            PanelGate::evaluate(Tab::Record, &ConnectState::Connected, true),
            PanelGate::Blocked(GateReason::SyncLost)
        );
    }

    #[test]
    fn not_connected_outranks_sync_lost() {
        assert_eq!(
            PanelGate::evaluate(Tab::Record, &ConnectState::Disconnected, true),
            PanelGate::Blocked(GateReason::NotConnected)
        );
    }

    #[test]
    fn failed_connection_blocks_like_not_connected() {
        assert_eq!(
            PanelGate::evaluate(Tab::Record, &ConnectState::Failed("no camera".into()), false),
            PanelGate::Blocked(GateReason::NotConnected)
        );
    }

    #[test]
    fn reason_messages_point_to_where_to_fix_it() {
        assert!(GateReason::NotConnected.message().contains("相机"));
        assert!(GateReason::SyncLost.message().contains("相机"));
        assert!(!GateReason::Connecting.message().is_empty());
    }
}
