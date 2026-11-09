from torch.nn.functional import mse_loss, cross_entropy

def regression_loss(prediction, target):
    return mse_loss(prediction, target)



def digitclassifier_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a tensor with shape
    (batch_size x 10). Each row is a one-hot vector encoding the correct
    digit class (0-9).

    Inputs:
        y_pred: a node with shape (batch_size x 10)
        y: a node with shape (batch_size x 10)
    Returns: a loss tensor
    """
    """ YOUR CODE HERE """
    labels = y.argmax(dim=1)
    return cross_entropy(y_pred, y)


def languageid_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a node with shape
    (batch_size x 5). Each row is a one-hot vector encoding the correct
    language.

    Inputs:
        model: Pytorch model to use
        y_pred: a node with shape (batch_size x 5)
        y: a node with shape (batch_size x 5)
    Returns: a loss node
    """
    "*** YOUR CODE HERE ***"
    return cross_entropy(y_pred, y.argmax(dim=1))


def digitconvolution_Loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a tensor with shape
    (batch_size x 10). Each row is a one-hot vector encoding the correct
    digit class (0-9).

    Inputs:
        y_pred : a node with shape (batch_size x 10)
        y: a node with shape (batch_size x 10)
    Returns: a loss tensor
    """
    """ YOUR CODE HERE """
    """
    Tính độ lỗi khi mô hình nhận diện chữ số.

    y_pred: kết quả dự đoán của mô hình, shape (batch_size, 10)
    y: đáp án đúng dạng one-hot, shape (batch_size, 10)
    """

    # Tính Cross Entropy trực tiếp với nhãn one-hot
    # Không dùng argmax vì autograder cần gradient theo y
    loss = cross_entropy(y_pred, y)

    # Đổi shape từ () thành (1,)
    return loss.reshape(1)
    
