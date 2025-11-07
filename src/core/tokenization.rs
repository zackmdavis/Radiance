use std::collections::{BTreeMap, HashMap};

use log::{info, warn};

pub const STANDARD_VOCABULARY: [char; 97] = [
    '▶', // start of sequence
    '\n', ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C',
    'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|',
    '}', '—',
];

#[derive(Debug)]
pub struct TokenVocabulary {
    pub token_to_id: HashMap<String, u16>,
    pub id_to_token: HashMap<u16, String>,

    pub(super) merge_rules: Vec<(String, String)>,
}

impl TokenVocabulary {
    pub fn new(tokens: Vec<char>) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        for (i, token) in tokens.iter().enumerate() {
            id_to_token.insert(i as u16, (*token).to_string());
            token_to_id.insert((*token).to_string(), i as u16);
        }
        TokenVocabulary {
            token_to_id,
            id_to_token,
            merge_rules: Vec::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.token_to_id.len() // without loss of generality
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        for (_i, merge_rule) in self.merge_rules.iter().enumerate() {
            let mut revised_tokens = Vec::new();
            let mut skip_next = false;
            for bigram in tokens.windows(2) {
                if skip_next {
                    skip_next = false;
                    continue;
                }
                let [first, second] = bigram else {
                    unreachable!();
                };
                if *first == merge_rule.0 && *second == merge_rule.1 {
                    revised_tokens.push(first.to_owned() + second);
                    skip_next = true;
                } else {
                    revised_tokens.push(first.to_owned());
                }
            }
            if !skip_next {
                revised_tokens.push(tokens.last().unwrap().to_owned());
            }
            tokens = revised_tokens;
        }
        tokens
    }

    pub fn token_id_ize(&self, text: &str) -> Vec<f32> {
        let tokens = self.tokenize(text);
        let mut ids = Vec::new();
        for token in tokens {
            if let Some(token_id) = self.token_to_id.get(&token) {
                ids.push(*token_id as f32);
            } else {
                warn!("ignoring unknown token {}", token);
            }
        }
        ids
    }

    pub fn new_from_corpus(training_megastring: String, vocabulary_size: u16) -> Self {
        let mut training_tokens: Vec<String> =
            training_megastring.chars().map(|c| c.to_string()).collect();
        let mut merge_rules: Vec<(String, String)> = Vec::new();
        // u16 range is from 0–65,535; our base alphabet is 97 chars, so we
        // have at most 65536−97 = 65439 other tokens to learn.
        //
        // We don't actually want to learn that many, because a larger
        // vocabulary means spending more parameters on the embedding matrix.
        for _iteration in 0..vocabulary_size - 97 {
            let mut bigram_counter = BTreeMap::<(String, String), usize>::new();
            for bigram in training_tokens.windows(2) {
                let [first, second] = bigram else {
                    unreachable!();
                };
                *bigram_counter
                    .entry((first.to_owned(), second.to_owned()))
                    .or_insert(0) += 1;
            }
            let (merge, _count) = bigram_counter
                .iter()
                .max_by_key(|&(ref _bigram, &count)| count)
                .unwrap();
            let mergetoken = merge.0.clone() + &merge.1;
            info!("initializing vocabulary: learned token {:?}", mergetoken);
            merge_rules.push(merge.clone());

            let mut revised_training_tokens = Vec::new();
            let mut skip_next = false;
            for bigram in training_tokens.windows(2) {
                if skip_next {
                    skip_next = false;
                    continue;
                }
                let [first, second] = bigram else {
                    unreachable!();
                };
                if *first == merge.0 && *second == merge.1 {
                    revised_training_tokens.push(mergetoken.clone());
                    skip_next = true;
                } else {
                    revised_training_tokens.push(first.to_owned());
                }
            }
            if !skip_next {
                revised_training_tokens.push(training_tokens.last().unwrap().to_owned());
            }
            training_tokens = revised_training_tokens
        }

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in STANDARD_VOCABULARY.iter().enumerate() {
            token_to_id.insert(token.to_string(), id as u16);
            id_to_token.insert(id as u16, token.to_string());
        }
        for (rule_id, merge) in merge_rules.iter().enumerate() {
            let token = merge.0.clone() + &merge.1;
            token_to_id.insert(
                token.to_owned(),
                (rule_id + STANDARD_VOCABULARY.len()) as u16,
            );
            id_to_token.insert(
                (rule_id + STANDARD_VOCABULARY.len()) as u16,
                token.to_owned(),
            );
        }

        TokenVocabulary {
            token_to_id,
            id_to_token,
            merge_rules,
        }
    }
}

impl Default for TokenVocabulary {
    fn default() -> Self {
        Self::new(STANDARD_VOCABULARY.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenization() {
        let mock_megastring = "She had promised to marry a scientist! It was too overwhelming a thought to entertain standing there by the window. She sought the room's most comfortable chair and braced herself to the situation.

If, one month before, a gossiping daughter of Fate had come to her with—\"Shall I tell you something?—_You_ are going to marry a man of science!\"—she would have smiled serenely at Fate's amusing mistake and responded—\"My good friend, it is quite true that great uncertainty attends this subject. So much to be expected is the unexpected, that I am quite willing to admit I _may_ marry the hurdy-gurdy man who plays beneath my window. I know life well enough to appreciate that I _may_ marry a pawnbroker or the Sultan of Turkey. I assert but one thing. I shall _not_ marry a 'man of science.'\"

And now, not only had she promised to marry a man of science, but she had quite overlooked the fact of his being one! And the thing which stripped her of the last shred of consistency was that she was to marry, not the every-day, average \"man of science,\" but one of the foremost scientists of all the world! The powers in charge of things matrimonial must be smiling a quiet little smile to-night.

But ah—here was the vindication! He had not _asked_ her to marry him. He had simply come and told her she _was_ to marry him. And he was a great, strong man—far more powerful than she. She had had positively nothing to do with it! Was it _her_ fault that he chanced to be engaged in scientific pursuits? And when he took her face so tenderly in his two hands—looked so far down into her eyes—and told her in a voice she would follow to the ends of the earth that he _loved_ her—was there any time then to think of paltry non-essentials like art and science?

But she thought of them a little now. How could she get away from them when each year of her past marched slowly in front of her, paused for an instant that she might get a full view, and then passed grinningly back to the abyss of things gone, from over the shoulder tossing straight into her consciousness a jeering, deep sinking \"_You too?_\"

Ernestine Stanley—that was the name she read in one of her books open beside her. Why her very _name_ stood for that quarrel which had rent all the years!";
        let vocabulary = TokenVocabulary::new_from_corpus(mock_megastring.to_owned(), 125);
        assert_eq!(
            vocabulary.merge_rules,
            vec![
                ("e", " "),
                (" ", "t"),
                ("t", " "),
                ("h", "e "),
                ("i", "n"),
                ("e", "r"),
                ("d", " "),
                ("h", "a"),
                ("e", "n"),
                (" ", "a"),
                (" t", "o"),
                (" ", "m"),
                ("o", "f"),
                ("h", "er"),
                (" t", "he "),
                ("in", "g"),
                ("o", "n"),
                ("a", "r"),
                (" ", "s"),
                ("y", " "),
                ("a", "n"),
                ("a", "s"),
                ("o", "u"),
                (" ", "of"),
                ("t", "h"),
                ("o", "w"),
                ("i", "s"),
                ("e", "d ")
            ]
            .into_iter()
            .map(|pair| (pair.0.to_owned(), pair.1.to_owned()))
            .collect::<Vec<_>>()
        );
        let tokens = vocabulary.tokenize(mock_megastring);
        assert_eq!(
            tokens[..15],
            vec![
                "S", "he ", "ha", "d ", "p", "r", "o", "m", "is", "e", "d", " to", " m", "ar", "r",
            ]
        );
    }
}
