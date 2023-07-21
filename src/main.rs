use std::{
    collections::BTreeMap,
    env,
    fmt::{Display, Error, Formatter},
};

#[derive(Debug)]
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

fn all_paths(sz: usize) -> Vec<Path> {
    fn helper(sz: usize, cur: &mut Vec<usize>, out: &mut Vec<Path>) {
        if cur.len() == sz - 1 {
            out.push(Path {
                partition: cur.clone(),
            });
            return;
        }
        let i = cur.len();
        let lim = cur.last().cloned().unwrap_or(sz).min(sz - i - 1);
        for h in 0..=lim {
            cur.push(h);
            helper(sz, cur, out);
            cur.pop();
        }
    }

    let mut ret = vec![];
    helper(sz, &mut vec![], &mut ret);
    ret
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let sz: usize = args[1].parse().unwrap();

    let mut by_total_and_area = BTreeMap::<usize, BTreeMap<usize, Vec<Path>>>::new();

    for p in all_paths(sz).into_iter() {
        let a = p.area();
        let b = p.bounce();
        by_total_and_area
            .entry(a + b)
            .or_default()
            .entry(a)
            .or_default()
            .push(p);
    }

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
