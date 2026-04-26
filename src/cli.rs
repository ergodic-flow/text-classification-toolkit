use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "text-toolkit")]
#[command(about = "Text classification toolkit")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Train(TrainArgs),
    Predict(PredictArgs),
    Evaluate(EvaluateArgs),
    Repl(ReplArgs),
    Calibrate(CalibrateArgs),
}

#[derive(Clone, Debug, ValueEnum)]
pub enum ModelType {
    Binary,
    Multinomial,
    Ova,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum Solver {
    Lbfgs,
    Sgd,
}

#[derive(Clone, Debug, Default, ValueEnum)]
pub enum NgramRange {
    #[default]
    Unigrams,
    UnigramsBigrams,
    Bigrams,
}

#[derive(Args)]
pub struct ModelArgs {
    #[arg(short, long)]
    pub input: String,

    #[arg(short = 't', long, value_enum, default_value = "binary")]
    pub model_type: ModelType,

    #[arg(short, long, value_enum, default_value = "lbfgs")]
    pub solver: Solver,

    #[arg(short, long, value_enum, default_value = "unigrams")]
    pub ngrams: NgramRange,

    #[arg(long)]
    pub max_features: Option<usize>,

    #[arg(long)]
    pub class_weight: Option<String>,

    #[arg(long)]
    pub sample_weight: Option<String>,

    #[arg(short, long)]
    pub output: String,
}

#[derive(Args)]
pub struct TrainArgs {
    #[command(flatten)]
    pub model: ModelArgs,

    #[arg(long, default_value_t = 0.0)]
    pub l2_reg: f64,

    #[arg(long, default_value_t = 0.01)]
    pub sgd_learning_rate: f64,

    #[arg(long, default_value_t = 32)]
    pub sgd_batch_size: usize,

    #[arg(long, default_value_t = 1000)]
    pub max_iter: usize,

    #[arg(long, default_value_t = 10)]
    pub lbfgs_memory: usize,

    #[arg(long, default_value_t = 1e-6)]
    pub lbfgs_tol: f64,
}

#[derive(Args)]
pub struct PredictArgs {
    #[arg(short, long)]
    pub input: String,

    #[arg(short, long)]
    pub model: String,

    #[arg(short, long)]
    pub output: Option<String>,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum ReportFormat {
    Text,
    Json,
}

#[derive(Args)]
pub struct EvaluateArgs {
    #[arg(short, long)]
    pub input: String,

    #[arg(short, long)]
    pub model: String,

    #[arg(short, long)]
    pub report: Option<String>,

    #[arg(long, value_enum, default_value = "text")]
    pub format: ReportFormat,
}

#[derive(Args)]
pub struct ReplArgs {
    #[arg(short, long)]
    pub model: String,
}

#[derive(Args)]
pub struct CalibrateArgs {
    #[arg(short, long)]
    pub model: String,

    #[arg(short, long)]
    pub input: String,

    #[arg(short, long)]
    pub output: String,
}
