use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::BufWriter;

use hashbrown::HashMap;

pub struct Converter {}

impl Converter {

    // Creates a vocab file, edges file, and a features file
    pub fn dehydrate(graph: &str, delim: &str, features: Option<&str>) {
        let f = File::open(graph).expect("Cannot open graph file");
        let br = BufReader::new(f);

        // Output paths
        let vocab = File::create(format!("{}.vocab", graph))
            .expect("Cannot open vocab file for writing");
        let mut vocab = BufWriter::new(vocab);

        let ids   = File::create(format!("{}.features.id", graph))
            .expect("Cannot open id features file for writing");
        let mut ids = BufWriter::new(ids);

        let edges    = File::create(format!("{}.edges", graph))
            .expect("Cannot open edges file for writing");
        let mut edges = BufWriter::new(edges);

        let mut node_to_idx = HashMap::new();

        for line in br.lines() {
            let line = line.expect("Failed to read line!");

            let line = line.trim();
            if line.len() > 0 {
                let pieces: Vec<&str> = line.trim().split(delim).collect();
                if pieces.len() != 2 && pieces.len() != 3 {
                    eprintln!("Got {} pieces", pieces.len());
                    panic!(format!("Improper line: `{}`", line));
                }

                for idx in 0..2 {
                    if !node_to_idx.contains_key(pieces[idx]) {
                        let v_idx = node_to_idx.len();
                        node_to_idx.insert(pieces[idx].to_string(), v_idx);
                        write!(vocab, "{} {}\n", v_idx, pieces[idx])
                            .expect("Couldn't write vocab!");
                        write!(ids, "{} {}\n", v_idx, v_idx)
                            .expect("Couldn't write vocab!");
                    }
                }

                let w = if pieces.len() == 3 { pieces[2] } else { "1" };
                let f_node = node_to_idx[pieces[0]];
                let t_node = node_to_idx[pieces[1]];
                // Write out edge
                write!(edges, "{} {} {}\n", f_node, t_node, w)
                    .expect("Couldn't write out edge!");
            }
        }

        // Write out features, if they exist
        if let Some(f_path) = features {
            let f = File::open(f_path).expect("Cannot open graph file");
            let br = BufReader::new(f);

            let features = File::create(format!("{}.features", graph))
                .expect("Cannot open features file for writing");
            let mut features = BufWriter::new(features);

            for line in br.lines() {
                let line = line.expect("Failed to read line!");
                let pieces: Vec<&str> = line.trim().splitn(2, delim).collect();
                if let Some(idx) = node_to_idx.get(pieces[0]) {
                    // Write out
                    write!(features, "{} {}\n", idx, pieces[1])
                        .expect("Unable to write out features file");
                }
            }
        }
    }

    pub fn hydrate(embeddings: &str, vocab: &str, output: Option<&str>) {
        // Load up the vocab
        let f = File::open(vocab).expect("Cannot open graph file");
        let br = BufReader::new(f);
        let idx_to_vocab: HashMap<_,_> = br.lines().map(|l| {
            let l = l.unwrap();
            let pieces: Vec<_> = l.splitn(2, ' ').collect();
            let idx = pieces[0].parse::<usize>()
                .expect(&format!("Expected an int, received `{}`", pieces[0]));

            (idx, pieces[1].to_string())
        }).collect();

        let f = File::open(embeddings).expect("Cannot open graph file");
        let br = BufReader::new(f);

        let mut out: Box<dyn Write> = match output {
            Some(path) => {
                let fd = File::create(path).expect("Cannot open output file");
                Box::new(BufWriter::new(fd))
            },
            None => {
                Box::new(BufWriter::new(std::io::stdout()))
            }
        };

        for line in br.lines() {
            let line = line.unwrap();
            let line = line.trim();
            if line.len() == 0 {
                write!(out, "\n").unwrap();
            } else {
                let pieces: Vec<_> = line.splitn(2, ": ").collect();
                let id = pieces[0].parse::<usize>().unwrap();
                if let Err(err) = write!(out, "{}\t{}\n", idx_to_vocab[&id], pieces[1]) {
                    if err.kind() == std::io::ErrorKind::BrokenPipe {
                        // Someone closed stdout in all likelihood, just break
                        break
                    }
                    panic!("Failed to write to output: {:?}", err);
                }
            }
        }
    }

}
