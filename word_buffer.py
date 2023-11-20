def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def match_word(buffered_word, vocabulary, threshold=100):
    # Placeholder function for word matching
    # Implement your word matching logic using Levenshtein distance
    # Return the closest matching word from the vocabulary
    
    closest_match = None
    min_distance = float('inf')

    for word in vocabulary:
        distance = levenshtein_distance(buffered_word, word)

        if distance < min_distance and distance <= threshold:
            min_distance = distance
            closest_match = word

    return closest_match

stream = "aaaaaaaaabbbccc ddddeeefff gghhhiii jjjkkklll"
word_buffer = ""
words = []
vocabulary = ["abc", "def", "aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh", "iii", "jjj", "kkk", "lll"]

for char in stream:
    if char.isalpha():
        word_buffer += char
    elif char.isspace():
        # End of a word or space
        if word_buffer:
            # Attempt to match the buffered sequence to known words
            matched_word = match_word(word_buffer, vocabulary)
            words.append(matched_word if matched_word else "unkown")
            word_buffer = ""
    else:
        pass
        # Handle other characters if needed

# Add any remaining words
if word_buffer:
    matched_word = match_word(word_buffer, vocabulary)
    words.append(matched_word if matched_word else word_buffer)

print("Processed Words:", words)