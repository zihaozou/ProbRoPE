use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use crate::types::{EventBatch, RgbFrame, TriggerEvent};


pub enum EvkMsg { Events(EventBatch), Trigger(TriggerEvent), Eof }

pub enum FlirMsg { Frame(RgbFrame), Eof }

pub enum EvkRawMsg {
    Bytes(Vec<u8>),
    ResetState,
}

pub fn send_latest<T>(tx: &Sender<T>, rx: &Receiver<T>, msg: T) -> bool {
    match tx.try_send(msg) {
        Ok(()) => false,
        Err(TrySendError::Full(msg)) => {
            let _ = rx.try_recv();
            let _ = tx.try_send(msg);
            true
        }
        Err(TrySendError::Disconnected(_)) => false,
    }
}

pub fn preview_channel<T>(cap: usize) -> (Sender<T>, Receiver<T>) { bounded(cap) }
