# An example of using the viterbi algorithm to determine the most likely POS tag
# for a given sequence of words.

from learnz.ai.viterbi import viterbi

states = {"verb", "noun"}

# P(A | B) = transition_probabilities[B][A]
transition_probabilities = {
    None: {
        "verb": 0.25,
        "noun": 0.75
    },
    "verb": {
        "verb": 0.35,
        "noun": 0.55
    },
    "noun": {
        "verb": 0.5,
        "noun": 0.3
    }
}

# P(A | B) = emission_probabilities[B][A]
emission_probabilities = {
    "verb": {
        "california": 0.008,
        "seals": 0.02,
        "report": 0.06
    },
    "noun": {
        "california": 0.04,
        "seals": 0.05,
        "report": 0.07
    }
}

observations = ["california", "seals", "report"]
most_likely_sequence = viterbi(observations, states, transition_probabilities, emission_probabilities)

padding_length = max(map(lambda word: len(word), observations)) + 2
for word, tag in zip(observations, most_likely_sequence):
                   print(f"{word:{padding_length}}{tag}")
