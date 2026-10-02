use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {

    Replay {
        #[arg(long)]
        raw: String,
        #[arg(long)]
        avi: String,
        #[arg(long, default_value_t = 2.0)]
        fps: f64,
    },

    Live {
        #[arg(long, default_value_t = 30.0)]
        fps: f64,
        #[arg(long, default_value_t = 5000)]
        exposure_us: i64,
        #[arg(long, default_value_t = 10.0)]
        gain_db: f64,

        #[arg(long, default_value_t = 1)]
        trigger_polarity: i8,
    },
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Replay { raw, avi, fps } => fs_app::app::run_replay(raw, avi, fps),
        Cmd::Live { fps, exposure_us, gain_db, trigger_polarity } => {
            fs_app::app::run_live(fps, exposure_us, gain_db, trigger_polarity)
        }
    }
}
