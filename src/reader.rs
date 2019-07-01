use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use super::{Games};

pub trait GameReader {
    fn next_set(&mut self) -> Result<Option<Vec<Games>>,ReadErr>;
}

// Reads all games as a single set
pub struct AllGames {
    paths: Vec<String>
}

impl AllGames {
    pub fn new(paths: Vec<String>) -> Self {
        AllGames { paths: paths }
    }
}

impl GameReader for AllGames {
    fn next_set(&mut self) -> Result<Option<Vec<Games>>,ReadErr> {
        if self.paths.is_empty() {
             Ok(None)
        } else {
            let mut v = Vec::new();
            for path in self.paths.drain(0..) {
                v.extend(read_edges(&path)?);
            }
            Ok(Some(v))
        }
    }
}

// Each sub game set is considered a completely different consideration
pub struct EachSetSeparate {
    paths: Vec<String>,
    buffer: Vec<Games>
}

impl EachSetSeparate {
    pub fn new(mut paths: Vec<String>) -> Self {
        // Put the paths in reverse so we can pop off the end
        paths.reverse();
        EachSetSeparate { paths: paths, buffer: vec![] }
    }
}

impl GameReader for EachSetSeparate {
    fn next_set(&mut self) -> Result<Option<Vec<Games>>,ReadErr> {
        if !self.buffer.is_empty() {
            Ok(self.buffer.pop().map(|x| vec![x]))
        } else if !self.paths.is_empty() {
            // Get the next set from the path
            let path = self.paths.pop()
                .expect("Shouldn't have failed since we check emptiness above");

            let mut games = read_edges(&path)?;
            games.reverse();
            self.buffer.extend(games);
            self.next_set()
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug)]
pub enum ReadErr {
    BadFormat,
    BadRowFormat(String)
}

/// Reads a set of edges in from disk.  New lines designate new "batches" of games
/// which can be useful for certain rankers
fn read_edges(path: &str) -> Result<Vec<Games>,ReadErr> {
    let mut sets = Vec::new();
    let mut v = Vec::new();
    let f = File::open(path).expect("Error opening file");
    let br = BufReader::new(f);
    for line in br.lines() {
        let line = line
            .expect("Failed to read line!");

        if line.trim().len() == 0 && v.len() > 0 {
            v.shrink_to_fit();
            sets.push(v);
            // Create a new set
            v = Vec::new();
            continue
        }

        let pieces = line.split_whitespace();
        let p: Vec<&str> = pieces.collect();
        if p.len() < 2 {
            return Err(ReadErr::BadRowFormat(line.to_owned()))
        }
        
        let winner = p[0].parse::<u32>()
            .map_err(|_| ReadErr::BadFormat)?;
        let loser = p[1].parse::<u32>()
            .map_err(|_| ReadErr::BadFormat)?;

        let weight = if p.len() == 3 {
            p[2].parse::<f32>().map_err(|_| ReadErr::BadFormat)?
        } else {
            1.0
        };

        v.push((winner, loser, weight));
    }
    
    if v.len() > 0 {
        sets.push(v)
    }
    Ok(sets)
}


