def viterbi(observations, states, transition_probabilities, emission_probabilities):
    scores = []
    back_pointers = []

    # initialization
    probabilities = dict()
    pointers = dict()

    for state in states:
        probabilities[state] = transition_probabilities[None][state] * emission_probabilities[state][observations[0]]
        pointers[state] = None

    scores.append(probabilities)
    back_pointers.append(pointers)

    # walk forward
    for index in range(1, len(observations)):
        probabilities = dict()
        pointers = dict()

        for state in states:
            max_probability = None
            max_previous_state = None

            for previous_state in states:
                probability = transition_probabilities[previous_state][state] * scores[index - 1][previous_state]

                if max_probability is None or probability > max_probability:
                    max_probability = probability
                    max_previous_state = previous_state

            probabilities[state] = emission_probabilities[state][observations[index]] * max_probability
            pointers[state] = max_previous_state

        scores.append(probabilities)
        back_pointers.append(pointers)

    # rebuild sequence
    max_state = None
    for state, probability in scores[-1].items():
        if max_state is None or probability > scores[-1][max_state]:
            max_state = state
    
    states = []
    for offset in range(len(observations)):
        states.append(max_state)
        max_state = back_pointers[len(observations) - 1 - offset][max_state]

    return states[::-1]
