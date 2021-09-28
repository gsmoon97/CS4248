# you are not allowed to import any other packages
import torch


def question1(shape):
    """
    Define a function to return a zero tensor with the given shape
    """
    return torch.zeros(shape)


def question2(data):
    """
    Define a function to convert a python list into a pytorch tensor with pytorch float data type
    TIP: use pytorch data type, not python native data type
    """
    return torch.FloatTensor(data)


def question3(a, b):
    """
    Define a function to compute 2*a+b
    """
    return 2 * a + b


def question4(a):
    """
    Define a function to get the first column from a 2-dimensional tensor
    """
    return a[:, 0]


def question5(data):
    """
    Define a function to combine a list of tensors into a new tensor at
    the 0-th dimension, and transpose the 0-th dimension with the 1st dimension
    """
    return torch.transpose(torch.stack(data, dim=0), 0, 1)


def question6(data):
    """
    Define a function to combine a list of 1-D tensors with different lengths
    into a new tensor by padding the shorter tensors with 0 on the right side
    """
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True)


def question7(w, x, b):
    """
    Define a function that calculates w*x + b
    """
    return (w @ x) + b


def question8(w, x, b):
    """
    Define a function that calculates batch w*x + b

    DO NOT use loop, list comprehension, or any similar operations. 
    """
    return torch.bmm(w, x) + b


def question9(x):
    """
    Given a 3-D tensor x (b, n, m), calculate the mean
    along dimension 1 without accounting for the 0-values. 
    """
    return x.sum(dim=1) / x.count_nonzero(dim=1)


def question10(pairs):
    """
    Define a funtion that calculates the dot product of pairs of vectors.
    You can use the functions from previous questions.
    """
    return torch.tensor([torch.tensor(pair[0]) @ torch.tensor(pair[1]) for pair in pairs])


def main():
    q1_input = (2, 3)
    print('Q1 Example input: \n{}\n'.format(q1_input))
    q1 = question1(q1_input)
    print('Q1 example output: \n{}\n'.format(q1))
    q2_input = [[1, 2, 3], [4, 5, 6]]
    print('Q2 Example input: \n{}\n'.format(q2_input))
    q2 = question2(q2_input)
    print('Q2 example output: \n{}\n'.format(q2))
    print('Q3 Example input: \na: {}\nb: {}\n'.format(
        q2, question2([[1, 1, 1], [1, 1, 1]])))
    q3 = question3(q2, question2([[1, 1, 1], [1, 1, 1]]))
    print('Q3 example output: \n{}\n'.format(q3))
    print('Q4 Example input: \n{}\n'.format(q2))
    print('Q4 example output: \n{}\n'.format(question4(q2)))
    q5_input = [question4(q1), question4(q2), question4(q3)]
    print('Q5 Example input: \n{}\n'.format(q5_input))
    q5 = question5(q5_input)
    print('Q5 example output: \n{}\n'.format(q5))
    q6_input = [question2([1]), question2([2, 2]), question2([3, 3, 3])]
    print('Q6 Example input: \n{}\n'.format(q6_input))
    q6 = question6(q6_input)
    print('Q6 example output: \n{}\n'.format(q6))
    q7_input = (question2([[0.3, 0.2], [0.6, -0.1], [-0.3, 0.2]]), q2,
                question2([[0.02, -0.03, 0.01], [0.03, 0.02, -0.01], [0.02, 0, 0.01]]))
    print('Q7 Example input \nw: \n{}\nx: {}\nb: {}\n'.format(*q7_input))
    q7 = question7(*q7_input)
    print('Q7 example output: \n{}\n'.format(q7))
    q8_input = (question2([[[0.3, 0.2], [0.6, -0.1], [-0.3, 0.2]], [[-0.6, 0.5], [0.1, 0.2], [0.4, -0.2]]]),
                question2([[[1, 2, 3], [4, 5, 6]], [[2, 1, 2], [3, 2, 3]]]),
                question2([[[0.02, -0.03, 0.01], [0.03, 0.02, -0.01], [0.02, 0, 0.01]], [[0.01, -0.04, 0.0], [0.02, 0.01, -0.02], [0.01, -0.01, 0.02]]]))
    print('Q8 Example input \nw: \n{}\nx: {}\nb: {}\n'.format(*q8_input))
    q8 = question8(*q8_input)
    print('Q8 example output: \n{}\n'.format(q8))
    q9_input = question2([[[1.0, 1.2], [0., 0.], [0., 0.]], [[2.0, 2.2], [
                         2.2, 2.6], [0., 0.]], [[3.0, 3.2], [3.2, 3.4], [3.1, 3.6]]])
    print('Q9 Example input: \n{}\n'.format(q9_input))
    q9 = question9(q9_input)
    print('Q9 example output: \n{}\n'.format(q9))
    q10_input = [([1, 1, 1], [2, 2, 2]), ([1, 2, 3], [3, 2, 1]),
                 ([0.1, 0.2, 0.3], [0.33, 0.25, 0.1])]
    print('Q10 Example input: \n{}\n'.format(q10_input))
    q10 = question10(q10_input)
    print('Q10 example output: \n{}\n'.format(q10))

    print('\n==== A2 Part 1 Done ====')


if __name__ == "__main__":
    main()
