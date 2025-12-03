dataset_cleaned:
https://www.kaggle.com/datasets/ad2000x/dataset_cleaned
Include data without duplication and add index for each row.

extracted_features:
https://www.kaggle.com/datasets/ad2000x/extracted_features
Include extracted_features from dataset_cleaned.
# Feature Description

| Feature | Description |
|---------|-------------|
| `Index` | Unique identifier assigned during preprocessing (e.g., viki-000001, wiki-000001) |
| `ID` | Original document identifier from the dataset |
| `Name` | Source article or document name |
| `Sentence` | Raw sentence text |
| `Label` | Target label (0 = Simple/Vikidia, 1 = Complex/Wikipedia) |
| `LengthWords` | Number of words in the sentence |
| `LengthChars` | Number of characters in the sentence |
| `words_chars_ratio` | Ratio of word count to character count |
| `cos_simi` | Cosine similarity between sentence embedding and reference vector |
| `avg_word_len` | Average word length in characters |
| `long_word_ratio` | Proportion of words exceeding a length threshold |
| `ttr` | Type-Token Ratio (unique words / total words) |
| `punct_density` | Punctuation count normalized by sentence length |
| `comma_density` | Comma count normalized by sentence length |
| `digit_ratio` | Proportion of digit characters in the sentence |
| `upper_ratio` | Proportion of uppercase characters in the sentence |
| `has_parens` | Binary flag indicating presence of parentheses |
| `n_tokens` | Number of tokens identified by spaCy |
| `max_depth` | Maximum depth of the dependency parse tree |
| `avg_depth` | Average depth of tokens in the dependency parse tree |
| `avg_dependency_distance` | Mean distance between dependent tokens and their heads |
| `func_word_ratio` | Proportion of function words (determiners, prepositions, etc.) |
| `n_clauses` | Number of clauses detected in the sentence |
| `clause_ratio` | Ratio of clauses to total tokens |
| `noun_ratio` | Proportion of noun tokens |
| `verb_ratio` | Proportion of verb tokens |

data_with_features
https://www.kaggle.com/datasets/ad2000x/data-features/data
Include dataset_cleaned with extracted_features.

mislabel_candidates:
https://www.kaggle.com/datasets/ad2000x/mislabel-candidates
Include mislabel_candidates from 02_estimate_simplified_proportion.
