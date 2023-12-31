//! Option structs for CLI parsing.

use std::path::PathBuf;

use gumdrop::Options;

#[derive(Debug, Options)]
pub enum Opts {
    CalcTable(CalcTableOpts),
    DrawTable(DrawTableOpts),
    CalcTableRow(CalcTableRowOpts),
    CalcTableCell(CalcTableCellOpts),
    CalcAlmostMinimalLine(CalcAlmostMinimalLineOpts),
    ShowMinimal(ShowMinimalOpts),
    ShowAll(ShowAllOpts),
}

#[derive(Debug, Options)]
pub struct CalcTableOpts {
    #[options(free)]
    pub sz: usize,

    #[options(short = "o")]
    pub output: Option<PathBuf>,

    #[options(short = "f")]
    pub force: bool,
}

#[derive(Debug, Options)]
pub struct DrawTableOpts {
    #[options(free)]
    pub in_path: PathBuf,

    #[options(free)]
    pub out_path: PathBuf,
}

#[derive(Debug, Options)]
pub struct CalcTableRowOpts {
    #[options(free)]
    pub sz: usize,

    #[options(free)]
    pub area: Option<usize>,
}

#[derive(Debug, Options)]
pub struct CalcTableCellOpts {
    #[options(free)]
    pub len: usize,

    #[options(free)]
    pub area: usize,

    #[options(free)]
    pub bounce: usize,
}

#[derive(Debug, Options)]
pub struct CalcAlmostMinimalLineOpts {
    #[options(free)]
    pub n: usize,
}

#[derive(Debug, Options)]
pub struct ShowMinimalOpts {
    #[options(free)]
    pub start: usize,

    #[options(free)]
    pub end: usize,
}

#[derive(Debug, Options)]
pub struct ShowAllOpts {
    #[options(free)]
    pub sz: usize,
}
