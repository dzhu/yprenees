use std::{
    cmp::Reverse,
    collections::BTreeMap,
    fmt::{Display, Error, Formatter},
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

mod draw;

#[derive(Debug, Options)]
enum Opts {
    DrawTable(DrawTableOpts),
    CountMinimal(CountMinimalOpts),
    ShowAll(ShowAllOpts),
}

#[derive(Debug, Options)]
struct DrawTableOpts {
    #[options(free)]
    sz: usize,
}

#[derive(Debug, Options)]
struct CountMinimalOpts {
    #[options(free)]
    start: usize,

    #[options(free)]
    end: usize,
}

#[derive(Debug, Options)]
struct ShowAllOpts {
    #[options(free)]
    sz: usize,
}

#[derive(Clone, Debug)]
struct Path {
    partition: Vec<usize>,
}

impl Path {
    fn area(&self) -> usize {
        let sz = self.partition.len();
        sz * (sz + 1) / 2 - self.partition.iter().sum::<usize>()
    }

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

fn for_all_paths<F: FnMut(&Path)>(sz: usize, cb: &mut F) {
    fn helper<F: FnMut(&Path)>(sz: usize, cur: &mut Vec<usize>, cb: &mut F) {
        if cur.len() == sz - 1 {
            let path = Path {
                partition: mem::take(cur),
            };
            cb(&path);
            let Path {
                partition: mut path,
            } = path;
            mem::swap(&mut path, cur);
            return;
        }
        let i = cur.len();
        let lim = cur.last().cloned().unwrap_or(sz).min(sz - i - 1);
        for h in 0..=lim {
            cur.push(h);
            helper(sz, cur, cb);
            cur.pop();
        }
    }

    helper(sz, &mut vec![], cb);
}

fn calc_partitions(end: usize) -> Vec<usize> {
    // nums[n][k] is the number of partitions of n with largest part k.
    let mut nums = vec![vec![1]];
    for n in 1..=end {
        nums.push(
            (0..=n)
                .map(|k| {
                    if k == 0 {
                        0
                    } else {
                        (0..=k).map(|j| nums[n - k].get(j).unwrap_or(&0)).sum()
                    }
                })
                .collect(),
        );
    }
    nums.into_iter()
        .map(|row| row.into_iter().sum::<usize>())
        .collect()
}

fn tri(n: usize) -> usize {
    n * (n + 1) / 2
}

fn search_minimal_partitions(sz: usize) -> Vec<Vec<usize>> {
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
    let mut best_total = sz * (sz + 1) / 2;
    for start in (1..=sz).rev() {
        helper(
            &mut vec![start],
            tri(start - 1) + sz - start,
            &mut best_total,
            sz - start,
            start,
            1,
            &mut |p| {
                ret.push(p.to_owned());
            },
        );
    }
    ret
}

fn count_minimal_partitions(start: usize, end: usize) {
    for sz in start..=end {
        fn value(p: &[usize]) -> usize {
            let a: usize = p.iter().map(|&n| n * (n - 1) / 2).sum();
            let b: usize = p.iter().enumerate().map(|(i, &n)| i * n).sum();
            a + b
        }

        let ps = search_minimal_partitions(sz);
        let min = ps.iter().map(|p| value(p)).min().unwrap();
        println!("================ {sz} {min} {}", ps.len());
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
            if true {
                println!(
                    "\x1b[{}m{} {p:?}\x1b[m",
                    if v == min { "32;1" } else { "" },
                    v
                );
            }
        }
        println!("{num_mins} mins");
        for (area, num) in areas {
            println!("{area} {num}");
        }
    }
}

fn draw_ab_counts(sz: usize) -> RgbImage {
    const BOX_SEP: usize = 22;

    let mut by_area_and_bounce = BTreeMap::<usize, BTreeMap<usize, usize>>::new();
    for_all_paths(sz, &mut |p| {
        let a = p.area();
        let b = p.bounce();
        *by_area_and_bounce
            .entry(a)
            .or_default()
            .entry(b)
            .or_default() += 1;
    });

    let max = tri(sz - 1);
    let partitions = calc_partitions(max);
    let img_dim = (BOX_SEP * (max + 1) + 1) as u32;

    let mut img = RgbImage::new(img_dim, img_dim);

    let line_color = [64, 64, 64].into();
    let text_color = [255, 255, 255].into();
    let text_style = TextStyle::new(&tamzen::FONT_6x12, BinaryColor::On);

    let mut min_locs = vec![];
    for area in 0..=max {
        for bounce in 0..=max - area {
            if let Some(&n) = by_area_and_bounce.get(&area).and_then(|m| m.get(&bounce)) {
                let is_chain_start = area == 0
                    || *by_area_and_bounce
                        .get(&(area - 1))
                        .and_then(|m| m.get(&(bounce + 1)))
                        .unwrap_or(&0)
                        != n
                    || bounce == 0
                    || *by_area_and_bounce
                        .get(&(area + 1))
                        .and_then(|m| m.get(&(bounce - 1)))
                        .unwrap_or(&0)
                        != n;
                let is_partition = n == partitions[max - (bounce + area)];

                let box_color = match (is_chain_start, is_partition) {
                    (true, true) => [0, 70, 100],
                    (true, false) => [0, 50, 0],
                    (false, true) => [0, 0, 80],
                    (false, false) => [0, 0, 0],
                };

                drawing::draw_filled_rect_mut(
                    &mut img,
                    Rect::at((BOX_SEP * area) as _, (BOX_SEP * bounce) as _)
                        .of_size(BOX_SEP as _, BOX_SEP as _),
                    box_color.into(),
                );

                let s = format!(
                    "{}",
                    n as isize - partitions[max - (bounce + area)] as isize
                );
                let x = (BOX_SEP * area + BOX_SEP / 2) as i32 + 1;
                let y = (BOX_SEP * bounce + BOX_SEP / 2) as i32 + 1;
                let pos = Point::new(x, y);
                let metrics = text_style.measure_string(&s, pos, Baseline::Middle);
                Text::new(&s, pos - metrics.bounding_box.size / 2, text_style)
                    .draw(&mut draw::ImageDrawTargetWrapper::new(&mut img, text_color))
                    .unwrap();

                if area == 0
                    || by_area_and_bounce
                        .get(&(area - 1))
                        .and_then(|m| m.get(&bounce))
                        .is_none()
                    || bounce == 0
                    || by_area_and_bounce
                        .get(&area)
                        .and_then(|m| m.get(&(bounce - 1)))
                        .is_none()
                {
                    min_locs.push((area, bounce));
                }
            }
        }
    }

    for i in 0..=max + 1 {
        let p0 = ((BOX_SEP * i) as f32, 0.0);
        let p1 = ((BOX_SEP * i) as f32, (BOX_SEP * (max + 2 - i)) as f32);
        drawing::draw_line_segment_mut(&mut img, p0, p1, line_color);
        drawing::draw_line_segment_mut(&mut img, (p0.1, p0.0), (p1.1, p1.0), line_color);
    }

    min_locs.sort_by_key(|&(a, b)| (a, Reverse(b)));
    for (a, b) in min_locs {
        println!("{a} {b}");
    }

    img
}

fn main() {
    let opts = Opts::parse_args_default_or_exit();

    match opts {
        Opts::DrawTable(DrawTableOpts { sz }) => {
            let img = draw_ab_counts(sz);
            img.save("/tmp/peaks.png").unwrap();
            img.save(format!("/tmp/peaks{sz:02}.png")).unwrap();
        }
        Opts::CountMinimal(CountMinimalOpts { start, end }) => {
            count_minimal_partitions(start, end);
        }
        Opts::ShowAll(ShowAllOpts { sz }) => {
            let mut by_total_and_area = BTreeMap::<usize, BTreeMap<usize, Vec<Path>>>::new();

            for_all_paths(sz, &mut |p| {
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
    }
}
