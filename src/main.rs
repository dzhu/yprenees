use std::{
    cmp::Reverse,
    collections::BTreeMap,
    fmt::{Display, Error, Formatter},
    fs::{self, File},
    mem,
};

use bitmap_font::{tamzen, TextStyle};
use embedded_graphics::{
    pixelcolor::BinaryColor,
    prelude::Point,
    text::{renderer::TextRenderer, Baseline, Text},
    Drawable,
};
use gumdrop::Options;
use image::RgbImage;
use imageproc::{drawing, rect::Rect};

mod color_maps;
mod draw;
mod opts;

/// A Dyck path, equipped with the ability to compute relevant statistics.
#[derive(Clone, Debug)]
struct Path {
    /// The partition for which the unfilled space between the path and the
    /// maximum-area path is the Young diagram (with parts read going up-right,
    /// in the view where the steps are up-right and down-right).
    ///
    /// The Dyck path is of order one greater than the length of this partition,
    /// since the last move in the path would always correspond to a zero in the
    /// partition, so we just leave out that zero.
    partition: Vec<usize>,
}

impl Path {
    /// Computes the area of this path (the number of full squares underneath
    /// it).
    fn area(&self) -> usize {
        tri(self.partition.len()) - self.partition.iter().sum::<usize>()
    }

    /// Computes the bounce locations of this path, measured from the right end.
    fn bounce_locs(&self) -> Vec<usize> {
        let n = self.partition.len() + 1;

        let mut ret = vec![];
        let mut i = 0;
        while i < n - 1 && self.partition[i] > 0 {
            ret.push(self.partition[i]);
            i = n - self.partition[i];
        }
        ret
    }

    /// Computes the sum of bounce locations of this path.
    fn bounce(&self) -> usize {
        let n = self.partition.len() + 1;

        let mut b = 0;
        let mut i = 0;
        while i < n - 1 {
            b += self.partition[i];
            i = n - self.partition[i];
        }
        b
    }
}

impl Display for Path {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let n = self.partition.len();
        for y in 0..n + 1 {
            if y > 0 {
                writeln!(f)?;
            }
            for (x, &n) in self.partition.iter().take(n - y).enumerate() {
                let c = match (n <= y, (x + y) % 2 == 0) {
                    (true, true) => 253,
                    (true, false) => 251,
                    (false, true) => 235,
                    (false, false) => 234,
                };
                write!(f, "\x1b[48;5;{c}m  \x1b[m")?;
            }
            write!(f, "\x1b[30m\x1b[48;5;58m \u{2584}\x1b[m")?;
        }
        Ok(())
    }
}

/// Calls `cb` on a [`Path`] containing the given partition.
fn call_path_cb<F: FnMut(&Path)>(part: &mut Vec<usize>, cb: &mut F) {
    let mut path = Path {
        partition: mem::take(part),
    };
    cb(&path);
    mem::swap(&mut path.partition, part);
}

/// Calls `cb` on each [`Path`] of the given length.
fn for_all_paths<F: FnMut(&Path)>(len: usize, cb: &mut F) {
    fn helper<F: FnMut(&Path)>(rem_cols: usize, last: usize, cur: &mut Vec<usize>, cb: &mut F) {
        if rem_cols == 0 {
            call_path_cb(cur, cb);
            return;
        }
        let lim = last.min(rem_cols);
        for h in 0..=lim {
            cur.push(h);
            helper(rem_cols - 1, h, cur, cb);
            cur.pop();
        }
    }

    helper(len - 1, len, &mut vec![], cb);
}

/// Calls `cb` on each [`Path`] of the given length and area.
fn for_paths_with_area<F: FnMut(&Path)>(len: usize, area: usize, cb: &mut F) {
    fn helper<F: FnMut(&Path)>(
        rem_cols: usize,
        last: usize,
        rem_area: usize,
        cur: &mut Vec<usize>,
        cb: &mut F,
    ) {
        if rem_cols == 0 {
            if rem_area == 0 {
                call_path_cb(cur, cb);
            }
            return;
        }
        let min = (0..).find(|n| n * rem_cols >= rem_area).unwrap();
        let max = last.min(rem_cols).min(rem_area);
        for h in min..=max {
            cur.push(h);
            helper(rem_cols - 1, h, rem_area - h, cur, cb);
            cur.pop();
        }
    }

    helper(len - 1, len, tri(len - 1) - area, &mut vec![], cb);
}

/// Calls `cb` on each [`Path`] of the given length, area, and bounce.
fn for_paths_with_area_and_bounce<F: FnMut(&Path)>(
    len: usize,
    area: usize,
    bounce: usize,
    cb: &mut F,
) {
    fn min_top_bounce(bounce: usize) -> usize {
        (0..).find(|&i| tri(i) >= bounce).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn helper<F: FnMut(&Path)>(
        sz: usize,
        last: usize,
        rem_area: usize,
        next_bounce_ind: usize,
        rem_bounce: usize,
        bounce_min: usize,
        cur: &mut Vec<usize>,
        cb: &mut F,
    ) {
        let rem_cols = sz - 1 - cur.len();
        if rem_cols == 0 {
            if rem_area == 0 && rem_bounce == 0 {
                call_path_cb(cur, cb);
            }
            return;
        }
        let min = (bounce_min..)
            .find(|n| n * rem_cols >= rem_area + tri(n.wrapping_sub(1)))
            .unwrap();
        let mut max = last.min(rem_cols).min(rem_area);

        let is_bounce = cur.len() == next_bounce_ind;
        if is_bounce {
            max = max.min(rem_bounce);
        }

        for h in min..=max {
            let next_bounce_ind = if is_bounce { sz - h } else { next_bounce_ind };
            let rem_bounce = rem_bounce - if is_bounce { h } else { 0 };
            let bounce_min = if is_bounce && h > 0 {
                min_top_bounce(rem_bounce)
            } else {
                bounce_min
            };

            cur.push(h);
            helper(
                sz,
                h,
                rem_area - h,
                next_bounce_ind,
                rem_bounce,
                bounce_min,
                cur,
                cb,
            );
            cur.pop();
        }
    }

    helper(
        len,
        len,
        tri(len - 1) - area,
        0,
        bounce,
        min_top_bounce(bounce),
        &mut vec![],
        cb,
    );
}

/// Computes the number of partitions of each integer from 0 to `end` (inclusive).
fn partition_numbers(end: usize) -> Vec<u128> {
    // The implementation uses the recurrence given by the pentagonal number
    // theorem.
    fn penta(n: isize) -> usize {
        (n * (3 * n - 1) / 2) as _
    }

    let mut ret = vec![1];

    for n in 1..=end {
        let mut val = 0i128;
        for k in 2.. {
            let p = penta((k >> 1) * (if k % 2 == 0 { 1 } else { -1 }));
            if p > n {
                break;
            }
            val += ret[n - p] as i128 * if k & 2 == 0 { -1 } else { 1 };
        }
        assert!(val > 0);
        ret.push(val as u128);
    }
    ret
}

/// Calculates the number of partitions of each integer from 0 to `a` * `b` that
/// fit in an `a`-by-`b` box.
fn restricted_partition_numbers(a: usize, b: usize) -> Vec<usize> {
    // Compute the Gaussian binomial coefficient [a + b // a]_q, whose
    // polynomial coefficients are exactly the desired numbers of partitions,
    // using the q-analog of Pascal's rule.
    //
    // Each line is a diagonal line in Pascal's triangle parallel to the right
    // side, which allows us to compute one row fully from the previous one and
    // then throw out the old one.
    //
    // This could instead take advantage of the symmetry of the coefficients to
    // potentially do less work, but the code is simpler this way and it's
    // plenty fast already.
    let mut line = vec![vec![1]; a + 1];
    for _n_minus_k in 1..=b {
        let mut line2 = vec![vec![1]];
        for k in 1..=a {
            let right = &line[k];
            let left = &line2[k - 1];
            line2.push(
                (0..right.len() + k)
                    .map(|i| if i >= k { right[i - k] } else { 0 } + left.get(i).unwrap_or(&0))
                    .collect(),
            );
        }
        line = line2;
    }

    line.pop().unwrap()
}

/// Computes the nth triangular number.
fn tri(n: usize) -> usize {
    n * n.wrapping_add(1) / 2
}

/// Finds partitions of `n` that may correspond to paths of length `n` with
/// minimal area-plus-bounce.
///
/// The returned partitions are those that contain no numbers repeated more than
/// twice or adjacent numbers with difference greater than 2. All actually
/// minimal paths are guaranteed to be returned, but others will be returned as
/// well.
fn potentially_minimal_partitions(n: usize) -> Vec<Vec<usize>> {
    fn helper<F: FnMut(&[usize])>(
        cur: &mut Vec<usize>,
        cur_total: usize,
        best_total: &mut usize,
        remaining: usize,
        last: usize,
        last_count: usize,
        cb: &mut F,
    ) {
        if remaining == 0 {
            *best_total = cur_total.min(*best_total);
            cb(cur);
            return;
        }
        if cur_total > *best_total {
            return;
        }

        if last_count < 2 && last <= remaining {
            cur.push(last);
            helper(
                cur,
                cur_total + tri(last - 1) + remaining - last,
                best_total,
                remaining - last,
                last,
                last_count + 1,
                cb,
            );
            cur.pop();
        }
        for next in (last.saturating_sub(2).max(1)..last).rev() {
            if next <= remaining {
                cur.push(next);
                helper(
                    cur,
                    cur_total + tri(next - 1) + remaining - next,
                    best_total,
                    remaining - next,
                    next,
                    1,
                    cb,
                );
                cur.pop();
            }
        }
    }

    let mut ret = vec![];
    let mut best_total = tri(n);
    for start in (1..=n).rev() {
        helper(
            &mut vec![start],
            tri(start - 1) + n - start,
            &mut best_total,
            n - start,
            start,
            1,
            &mut |p| {
                ret.push(p.to_owned());
            },
        );
    }
    ret
}

/// Displays information about minimal paths of the given length in a
/// human-readable format.
fn show_minimal_partitions(start: usize, end: usize) {
    fn value(p: &[usize]) -> usize {
        let a: usize = p.iter().map(|&n| n * (n - 1) / 2).sum();
        let b: usize = p.iter().enumerate().map(|(i, &n)| i * n).sum();
        a + b
    }

    for len in start..=end {
        let ps = potentially_minimal_partitions(len);
        let min = ps.iter().map(|p| value(p)).min().unwrap();
        println!("================ len: {len}    min A+B: {min}");
        let mut num_mins = 0;
        let mut areas: BTreeMap<usize, usize> = BTreeMap::new();

        for p in ps {
            let v = value(&p);
            if v != min {
                continue;
            }
            num_mins += 1;
            let area: usize = p.iter().map(|n| tri(n - 1)).sum();
            *areas.entry(area).or_default() += 1;
            println!("{v} {p:?}");
        }
        println!("\x1b[32m{num_mins}\x1b[m minimal partitions");
        println!("counts by area:");
        for (area, num) in areas {
            println!("{area:3} {num:3}");
        }
    }
}

#[derive(Clone, Default)]
struct TwoDimMap<V> {
    vals: Vec<Vec<V>>,
}

impl<V: PartialEq<V>> PartialEq<TwoDimMap<V>> for TwoDimMap<V> {
    fn eq(&self, other: &TwoDimMap<V>) -> bool {
        self.vals == other.vals
    }
}

impl<V: Clone + Default + PartialEq<V>> TwoDimMap<V> {
    fn get_mut(&mut self, a: usize, b: usize) -> &mut V {
        if self.vals.len() <= a {
            self.vals.resize(a + 1, vec![]);
        }
        if self.vals[a].len() <= b {
            self.vals[a].resize(b + 1, Default::default());
        }
        &mut self.vals[a][b]
    }

    fn iter(&self) -> impl Iterator<Item = ((usize, usize), &V)> {
        self.vals
            .iter()
            .enumerate()
            .flat_map(|(a, row)| row.iter().enumerate().map(move |(b, val)| ((a, b), val)))
            .filter(|&(_, v)| !v.eq(&V::default()))
    }
}

impl<V> TwoDimMap<V> {
    fn clear(&mut self) {
        for row in &mut self.vals {
            row.clear();
        }
    }
}

impl<V> TwoDimMap<TwoDimMap<V>> {
    fn clear_inner(&mut self) {
        for row in &mut self.vals {
            for sub in row {
                sub.clear();
            }
        }
    }
}

/// Calculates the full area/bounce count table for paths of the given length.
fn calc_table(len: usize) -> Vec<Vec<u128>> {
    // Key: last column, next bounce location, area so far, bounce so far.
    let mut counts: TwoDimMap<TwoDimMap<u128>> = Default::default();
    *counts.get_mut(len - 1, 0).get_mut(0, 0) = 1;
    let mut counts2: TwoDimMap<TwoDimMap<u128>> = Default::default();

    for step in 0..len {
        for ((last_col, bounce_loc), sub) in counts.iter() {
            for ((area, bounce), count) in sub.iter() {
                for next_col in 0..=last_col.min(len - 1 - step) {
                    let next_area = area + len - 1 - step - next_col;
                    let next_bounce = bounce + if step == bounce_loc { next_col } else { 0 };
                    let next_bounce_loc = if step == bounce_loc {
                        len - next_col
                    } else {
                        bounce_loc
                    };
                    *counts2
                        .get_mut(next_col, next_bounce_loc)
                        .get_mut(next_area, next_bounce) += count;
                }
            }
        }
        counts.clear_inner();
        mem::swap(&mut counts, &mut counts2);
    }

    let max = tri(len - 1);
    let mut table: Vec<_> = (1..=max + 1).rev().map(|n| vec![0; n]).collect();

    for (_, sub) in counts.iter() {
        for ((area, bounce), count) in sub.iter() {
            table[area][bounce] += count;
        }
    }
    table
}

/// Performs a least-squares fit to the given points using a hyperbola in a
/// particular class.
///
/// The hyperbola is constrained to have axis-aligned asymptotes and its center
/// on the line x=y, so its equation is of the form (x - d) * (y - d) = c^2. The
/// return value is (c, d).
fn fit_hyperbola(pts: &[(f64, f64)]) -> (f64, f64) {
    let s2 = 1.0 / 2.0f64.sqrt();
    let n = pts.len() as f64;
    let pts: Vec<_> = pts
        .iter()
        .map(|&(x, y)| ((x - y) * s2, (x + y) * s2))
        .collect();

    let mut c = (pts.iter().map(|&(x, y)| y * y - x * x).sum::<f64>() / n).sqrt();
    let mut d = 0.0;

    for _ in 0..20000 {
        let diff_c = c * (n - pts.iter().map(|&(x, y)| (y - d) / c.hypot(x)).sum::<f64>());
        let delta_c = diff_c * 0.005;
        let d2 = pts.iter().map(|&(x, y)| y - c.hypot(x)).sum::<f64>() / n;

        if delta_c.abs() < 1e-6 && (d2 - d).abs() < 1e-6 {
            break;
        }
        c -= delta_c;
        d = d2;
    }
    (c * s2, d * s2)
}

/// Creates an image representing the given area/bounce count table.
fn draw_table(table: &[Vec<u128>]) -> RgbImage {
    const BOX_SEP: usize = 22;

    assert!(table
        .iter()
        .enumerate()
        .all(|(i, row)| i + row.len() == table.len()));
    let max = table.len() - 1;

    let img_dim = (BOX_SEP * (max + 1) + 1) as u32;

    let mut img = RgbImage::new(img_dim, img_dim);

    let line_color = [64, 64, 64].into();
    let text_color = [255, 255, 255].into();
    let text_style = TextStyle::new(&tamzen::FONT_5x9, BinaryColor::On);

    let mut min_locs = vec![];

    let partitions = partition_numbers(max);
    let sz = (1..).find(|n| tri(n - 1) >= max).unwrap();
    assert_eq!(tri(sz - 1), max);
    let restricted_partitions = if sz >= 3 {
        restricted_partition_numbers(sz - 3, sz - 2)
    } else {
        vec![]
    };

    // Draw cell contents.
    for area in 0..=max {
        for bounce in 0..=max - area {
            let n = table[area][bounce];
            if n == 0 {
                continue;
            }

            #[allow(clippy::collapsible_else_if)]
            let num_chains_started = n - if area > bounce {
                if bounce > 0 {
                    table[area + 1][bounce - 1]
                } else {
                    0
                }
            } else {
                if area > 0 {
                    table[area - 1][bounce + 1]
                } else {
                    0
                }
            };

            let correction_seq = [
                1, 2, 5, 9, 16, 26, 42, 64, 97, 142, 206, 292, 411, 568, 780, 1057, 1423, 1896,
                2512, 3299, 4311, 5593, 7222, 9269, 11846, 15059, 19070, 24039,
            ];

            let (num_chains_is_partition, num_chains_is_almost_partition) = {
                let (x, y) = (area.min(bounce), area.max(bounce));
                let y_end = max - 2 * x;
                (
                    y_end >= y && num_chains_started == partitions[y_end - y],
                    sz >= 4
                        && y_end >= y + sz - 4
                        && num_chains_started
                            == partitions[y_end - y]
                                - correction_seq.get(y_end - (y + sz - 4)).unwrap_or(&0),
                )
            };

            let is_chain_start = num_chains_started > 0;
            let is_restricted_partition = Some(n)
                == restricted_partitions
                    .get(max - (bounce + area))
                    .map(|&x| x as u128);
            if (area, bounce) == (max / 3 + 1, max / 3 + 1) {
                assert!(!is_chain_start);
            }

            let box_color = match (
                num_chains_is_almost_partition,
                num_chains_is_partition,
                is_chain_start,
                is_restricted_partition,
            ) {
                (true, _, _, _) => [100, 0, 100],
                (false, true, true, true) => [0, 80, 100],
                (false, true, true, false) => [0, 50, 0],
                (false, false, true, true) => [0, 70, 100],
                (false, false, true, false) => [80, 80, 0],
                (false, false, false, true) => [0, 0, 100],
                (false, false, false, false) => [0, 0, 0],
                _ => unreachable!(),
            };

            drawing::draw_filled_rect_mut(
                &mut img,
                Rect::at((BOX_SEP * area) as _, (BOX_SEP * bounce) as _)
                    .of_size(BOX_SEP as _, BOX_SEP as _),
                box_color.into(),
            );

            let s = format!("{}", n);
            let num_lines = (s.len() - 1) / 4 + 1;
            let line_len = (s.len() - 1) / num_lines + 1;

            let line_height = text_style
                .measure_string(&s, Point::new(0, 0), Baseline::Middle)
                .bounding_box
                .size
                .height
                - 2;
            let x = (BOX_SEP * area + BOX_SEP / 2) as i32 + 1;
            let y = (BOX_SEP * bounce + BOX_SEP / 2 - line_height as usize * (num_lines - 1) / 2)
                as i32
                + 1;

            for (i, line) in s.as_bytes().chunks(line_len).enumerate() {
                let pos = Point::new(x, y + line_height as i32 * i as i32);
                let line = std::str::from_utf8(line).unwrap();
                let metrics = text_style.measure_string(line, pos, Baseline::Middle);
                Text::new(line, pos - metrics.bounding_box.size / 2, text_style)
                    .draw(&mut draw::ImageDrawTargetWrapper::new(&mut img, text_color))
                    .unwrap();
            }

            if (area == 0 || table[area - 1][bounce] == 0)
                && (bounce == 0 || table[area][bounce - 1] == 0)
            {
                min_locs.push((area, bounce));
            }
        }
    }

    // Draw grid.
    for i in 0..=max + 1 {
        let p0 = ((BOX_SEP * i) as f32, 0.0);
        let p1 = ((BOX_SEP * i) as f32, (BOX_SEP * (max + 2 - i)) as f32);
        drawing::draw_line_segment_mut(&mut img, p0, p1, line_color);
        drawing::draw_line_segment_mut(&mut img, (p0.1, p0.0), (p1.1, p1.0), line_color);
    }

    min_locs.sort_by_key(|&(a, b)| (a, Reverse(b)));

    // Draw miniature heat map.
    let max_val = *table.iter().flat_map(|row| row.iter()).max().unwrap();
    if max_val > 1 {
        fn color_map(x: f32) -> [u8; 3] {
            color_maps::L16[((x * 255.0).round() as usize).clamp(0, 255)]
        }

        const SUB_SEP: usize = BOX_SEP / 2;
        let base = (BOX_SEP - SUB_SEP) * (max + 1) + 1;
        for (area, row) in table.iter().enumerate() {
            for (bounce, &n) in row.iter().enumerate() {
                if n == 0 {
                    continue;
                }
                let rel_val = (n as f32).log(max_val as f32);
                drawing::draw_filled_rect_mut(
                    &mut img,
                    Rect::at(
                        (base + (SUB_SEP * area)) as _,
                        (base + (SUB_SEP * bounce)) as _,
                    )
                    .of_size(SUB_SEP as _, SUB_SEP as _),
                    color_map(rel_val).into(),
                );
            }
        }
    }

    // Draw hyperbola fitted to boundary of table.
    let (c, d) = fit_hyperbola(
        &min_locs
            .iter()
            .map(|&(x, y)| (x as f64, y as f64))
            .collect::<Vec<_>>(),
    );

    for px in (0..=BOX_SEP * (max + 1)).rev() {
        let x = px as f64 / BOX_SEP as f64;
        let y = c * c / (x - d) + d;
        if y >= 0.0 {
            let py = (y * BOX_SEP as f64).round() as usize;
            img.put_pixel(px as u32, py as u32, [255, 255, 255].into());
            img.put_pixel(py as u32, px as u32, [255, 255, 255].into());

            if py >= px {
                break;
            }
        }
    }

    let text_style = TextStyle::new(
        if max >= 15 {
            &tamzen::FONT_12x24
        } else {
            &tamzen::FONT_5x9
        },
        BinaryColor::On,
    );
    let ss = [
        format!("semi major:  {:5.2}", c * 2.0f64.sqrt()),
        format!("center dist: {:5.2}", -d * 2.0f64.sqrt()),
    ];
    let mut y = 1;
    for s in ss {
        let pos = Point::new(1, y);
        let metrics = text_style.measure_string(&s, pos, Baseline::Middle);
        Text::new(&s, pos, text_style)
            .draw(&mut draw::ImageDrawTargetWrapper::new(&mut img, text_color))
            .unwrap();
        y += metrics.bounding_box.size.height as i32;
    }

    img
}

/// Displays information about all paths of the given length in a human-readable
/// format.
fn show_all(len: usize) {
    let mut by_total_and_area = BTreeMap::<usize, BTreeMap<usize, Vec<Path>>>::new();

    for_all_paths(len, &mut |p| {
        let a = p.area();
        let b = p.bounce();
        by_total_and_area
            .entry(a + b)
            .or_default()
            .entry(a)
            .or_default()
            .push(p.clone());
    });

    for (total, by_area) in by_total_and_area {
        println!("\x1b[32;1m================================================================ total: {total:2}\x1b[m");
        for (area, paths) in by_area.into_iter().rev() {
            println!(
                "\x1b[36m================ area: {area:2}  bounce: {:2}  ({:2} paths)\x1b[m",
                paths[0].bounce(),
                paths.len()
            );
            for p in paths {
                println!("bounces: {:?}", p.bounce_locs());
                println!("{p}");
            }
        }
    }
}

fn main() {
    use crate::opts::*;
    let opts = Opts::parse_args_default_or_exit();

    match opts {
        Opts::CalcTable(CalcTableOpts { sz, output, force }) => {
            use std::io::Write;
            let mut f: Box<dyn Write> = match output {
                Some(p) => Box::new(
                    File::options()
                        .write(true)
                        .create(true)
                        .create_new(!force)
                        .open(p)
                        .unwrap(),
                ),
                None => Box::new(std::io::stdout().lock()),
            };

            let t0 = std::time::Instant::now();
            let table = calc_table(sz);
            let dt = t0.elapsed();
            eprintln!("{sz} {t}", t = dt.as_secs_f64());

            writeln!(f, "{}", serde_json::to_string(&table).unwrap()).unwrap();
        }
        Opts::DrawTable(DrawTableOpts { in_path, out_path }) => {
            let table: Vec<Vec<u128>> =
                serde_json::from_slice(&fs::read(in_path).unwrap()).unwrap();
            draw_table(&table).save(out_path).unwrap();
        }
        Opts::CalcTableRow(CalcTableRowOpts { sz, area }) => {
            let area = area.unwrap_or(tri(sz - 1) / 3 + 1);
            let mut counts = vec![0; tri(sz - 1) - area + 1];
            for_paths_with_area(sz, area, &mut |p| counts[p.bounce()] += 1);
            let parts = partition_numbers(counts.len());

            let seq2 = [
                1, 3, 6, 11, 19, 31, 49, 75, 112, 164, 236, 334, 467, 645, 881, 1192, 1599, 2127,
                2809, 3684, 4801,
            ];

            let part_diff: Vec<_> = parts
                .iter()
                .zip(counts.iter().rev())
                .map(|(&p, &n)| p as isize - n as isize)
                .collect();

            let seq2_diff: Vec<_> = part_diff
                .iter()
                .skip(sz - 2)
                .zip(seq2.iter())
                .map(|(&p, &n)| p - n as isize)
                .collect();
            println!("{counts:?}");
            println!("\x1b[34m0 {:?}\x1b[m", part_diff);
            println!("\x1b[32m1 {:?}\x1b[m", seq2_diff);
        }
        Opts::CalcTableCell(CalcTableCellOpts { len, area, bounce }) => {
            let mut n = 0;
            for_paths_with_area_and_bounce(len, area, bounce, &mut |_| n += 1);
            println!("{len} {area} {bounce} {n}");
        }
        Opts::CalcAlmostMinimalLine(CalcAlmostMinimalLineOpts { n }) => {
            let len = tri(n);
            let tetra = (1..n).map(tri).sum::<usize>();
            let sum = 2 * tetra + 1;
            println!("sum: {sum}");
            for a in tetra + 1..=sum {
                let b = sum - a;
                let mut n = 0;
                let t0 = std::time::Instant::now();
                for_paths_with_area_and_bounce(len, a, b, &mut |_| n += 1);
                println!("{a} {b} {n} {:?}", t0.elapsed());
            }
        }

        Opts::ShowMinimal(ShowMinimalOpts { start, end }) => {
            show_minimal_partitions(start, end);
        }
        Opts::ShowAll(ShowAllOpts { sz }) => {
            show_all(sz);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    const EXHAUSTIVE_LIMIT: usize = 16;

    #[test]
    fn test_exhaustive() {
        for len in 1..EXHAUSTIVE_LIMIT {
            let table = calc_table(len);
            let max = tri(len - 1);

            // Check the shape and symmetry of the table in its own right.
            assert_eq!(table.len(), max + 1);
            for (a, row) in table.iter().enumerate() {
                assert_eq!(row.len(), max + 1 - a);
                for (b, _) in row.iter().enumerate() {
                    assert_eq!(table[b][a], table[a][b]);
                }
            }

            // Check per-row pruned search against the table.
            for (a, ref_row) in table.iter().enumerate() {
                let mut row = vec![0; max + 1 - a];
                for_paths_with_area(len, a, &mut |p| row[p.bounce()] += 1);
                assert_eq!(&row, ref_row);
            }

            // Check per-cell pruned search against the table.
            for (a, row) in table.iter().enumerate() {
                for (b, ref_count) in row.iter().enumerate() {
                    let mut count = 0;
                    for_paths_with_area_and_bounce(len, a, b, &mut |_| count += 1);
                    assert_eq!(&count, ref_count);
                }
            }
        }
    }

    #[test]
    fn test_row() {
        for len in EXHAUSTIVE_LIMIT..20 {
            let max = tri(len - 1);
            let area = max / 3 + 1;

            let mut row = vec![0; max + 1 - area];
            for_paths_with_area(len, area, &mut |p| row[p.bounce()] += 1);

            for (bounce, ref_count) in row.iter().enumerate() {
                let mut count = 0;
                for_paths_with_area_and_bounce(len, area, bounce, &mut |_| count += 1);
                assert_eq!(&count, ref_count);
            }
        }
    }

    #[test]
    fn test_restricted_partition() {
        fn restricted_partition_numbers_ref(a: usize, b: usize) -> Vec<usize> {
            let mut ret = vec![0; a * b + 1];
            fn helper(rem_cols: usize, last: usize, cur: usize, counts: &mut Vec<usize>) {
                if rem_cols == 0 {
                    counts[cur] += 1;
                    return;
                }
                for h in 0..=last {
                    helper(rem_cols - 1, h, cur + h, counts);
                }
            }
            helper(b, a, 0, &mut ret);
            ret
        }

        fn do_test(a: usize, b: usize) {
            let t0 = std::time::Instant::now();
            let ref_val = restricted_partition_numbers_ref(a, b);
            let ref_t = t0.elapsed();

            let t0 = std::time::Instant::now();
            let check_val = restricted_partition_numbers(a, b);
            let check_t = t0.elapsed();

            assert_eq!(ref_val, check_val, "{a} {b}");
            println!("{a} {b} {ref_t:?} {check_t:?}");
        }

        for a in 0..14 {
            for b in 0..14 {
                do_test(a, b);
            }
        }
        for a in 0..6 {
            for b in 0..60 {
                do_test(a, b);
            }
        }
    }

    #[test]
    fn test_calc_table() {
        fn calc_table_slow(len: usize) -> Vec<Vec<u128>> {
            let max = tri(len - 1);
            let mut table: Vec<_> = (1..=max + 1).rev().map(|n| vec![0; n]).collect();
            for_all_paths(len, &mut |p| {
                table[p.area()][p.bounce()] += 1;
            });
            table
        }

        for len in 1..=16 {
            println!("================ {len}");
            let t0 = std::time::Instant::now();
            let ref_val = calc_table_slow(len);
            let ref_t = t0.elapsed();

            let t0 = std::time::Instant::now();
            let check_val = calc_table(len);
            let check_t = t0.elapsed();
            assert_eq!(ref_val, check_val, "{len}");
            println!(
                "{ref_t:?} {check_t:?} {r}",
                r = check_t.as_secs_f64() / ref_t.as_secs_f64()
            );
        }
    }
}
