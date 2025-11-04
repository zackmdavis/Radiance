#![allow(dead_code)]

use std::collections::{HashMap, BTreeMap};

use super::embedding::{TokenVocabulary};

impl TokenVocabulary {
    pub fn new_from_corpus(training_megastring: String, vocabulary_size: u16) -> Self {
        let mut training_tokens: Vec<String> = training_megastring.chars().map(|c| c.to_string()).collect();
        let mut merge_rules: Vec<(String, String)> = Vec::new();
        // u16 range is from 0–65,535; our base alphabet is 97 chars, so we
        // have at most 65536−97 = 65439 other tokens to learn.
        //
        // We don't actually want to learn that many, because a larger
        // vocabulary means spending more parameters on the embedding matrix.
        for _iteration in 0..vocabulary_size-97 {
            let mut bigram_counter = BTreeMap::<(String, String), usize>::new();
            for bigram in training_tokens.windows(2) {
                let [first, second] = bigram else { unreachable!(); };
                *bigram_counter.entry((first.to_owned(), second.to_owned())).or_insert(0) += 1;
            }
            let (merge, _count) = bigram_counter.iter().max_by_key(|(_bigram, &count)| count).unwrap();
            let mergetoken = merge.0.clone() + &merge.1;
            merge_rules.push(merge.clone());

            let mut revised_training_tokens = Vec::new();
            let mut skip_next = false;
            for bigram in training_tokens.windows(2) {
                if skip_next {
                    skip_next = false;
                    continue;
                }
                let [first, second] = bigram else { unreachable!(); };
                if *first == merge.0 && *second == merge.1 {
                    revised_training_tokens.push(mergetoken.clone());
                    skip_next = true;
                } else {
                    revised_training_tokens.push(first.to_owned());
                }
            }
            training_tokens = revised_training_tokens
        }

        TokenVocabulary {
            merge_rules,

            // TODO: figure out how to set these
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenization_learning() {
        let mock_megastring = "She had promised to marry a scientist! It was too overwhelming a thought to entertain standing there by the window. She sought the room's most comfortable chair and braced herself to the situation.

If, one month before, a gossiping daughter of Fate had come to her with—\"Shall I tell you something?—_You_ are going to marry a man of science!\"—she would have smiled serenely at Fate's amusing mistake and responded—\"My good friend, it is quite true that great uncertainty attends this subject. So much to be expected is the unexpected, that I am quite willing to admit I _may_ marry the hurdy-gurdy man who plays beneath my window. I know life well enough to appreciate that I _may_ marry a pawnbroker or the Sultan of Turkey. I assert but one thing. I shall _not_ marry a 'man of science.'\"

And now, not only had she promised to marry a man of science, but she had quite overlooked the fact of his being one! And the thing which stripped her of the last shred of consistency was that she was to marry, not the every-day, average \"man of science,\" but one of the foremost scientists of all the world! The powers in charge of things matrimonial must be smiling a quiet little smile to-night.

But ah—here was the vindication! He had not _asked_ her to marry him. He had simply come and told her she _was_ to marry him. And he was a great, strong man—far more powerful than she. She had had positively nothing to do with it! Was it _her_ fault that he chanced to be engaged in scientific pursuits? And when he took her face so tenderly in his two hands—looked so far down into her eyes—and told her in a voice she would follow to the ends of the earth that he _loved_ her—was there any time then to think of paltry non-essentials like art and science?

But she thought of them a little now. How could she get away from them when each year of her past marched slowly in front of her, paused for an instant that she might get a full view, and then passed grinningly back to the abyss of things gone, from over the shoulder tossing straight into her consciousness a jeering, deep sinking \"_You too?_\"

Ernestine Stanley—that was the name she read in one of her books open beside her. Why her very _name_ stood for that quarrel which had rent all the years!";
        let vocabulary = TokenVocabulary::new_from_corpus(mock_megastring.to_owned(), 125);
        assert_eq!(
            vocabulary.merge_rules,
            vec! [("e", " "), (" ", "t"), ("t", " "), ("h", "e "), ("i", "n"), ("e", "r"), ("d", " "), ("h", "a"), ("e", "n"), (" ", "a"), (" t", "o"), (" ", "m"), ("o", "f"), ("h", "er"), ("in", "g"), (" t", "he "), ("o", "n"), (" ", "s"), ("a", "r"), ("y", " "), ("a", "n"), ("a", "s"), ("o", "u"), (" ", "of"), ("t", "h"), ("o", "w"), ("i", "s"), ("e", "d ")].into_iter().map(|pair| (pair.0.to_owned(), pair.1.to_owned())).collect::<Vec<_>>()
       )
    }
}
