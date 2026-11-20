from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""
def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        errors_per_epoch = []

        while True:
            mistakes = 0

            for batch in dataloader:
                x = batch['x']
                y = batch['label']

                if model.get_prediction(x) != y:
                    model.w += y * x
                    mistakes += 1

            errors_per_epoch.append(mistakes)

            if mistakes == 0:
                break

        return errors_per_epoch


"""
##################
### QUESTION 2 ###
##################
"""
def train_regression(model, dataset):
    """
    Train model to approximate sin(x).
    """
    dataloader = DataLoader(
        dataset,
        batch_size=50,
        shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(2000):
        model.train()

        for batch in dataloader:
            x = batch["x"]
            y = batch["label"]

            optimizer.zero_grad()
            prediction = model(x)
            loss = regression_loss(prediction, y)

            loss.backward()

            optimizer.step()

        model.eval()

        total_loss = 0.0
        total_samples = 0

        with no_grad():
            for batch in dataloader:
                x = batch["x"]
                y = batch["label"]

                prediction = model(x)
                loss = regression_loss(prediction, y)

                n = x.shape[0]
                total_loss += loss.item() * n
                total_samples += n

        average_loss = total_loss / total_samples

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {average_loss:.6f}")

      
        if average_loss <= 0.015:
            print("Training completed!")
            break



"""
##################
### QUESTION 3 ###
##################
"""
def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(10):
        for batch in dataloader:
            x = batch["x"]
            y = batch["label"]

            optimizer.zero_grad()
            y_pred = model(x)
            loss = digitclassifier_loss(y_pred, y)
            loss.backward()
            optimizer.step()


"""
##################
### QUESTION 4 ###
##################
"""
def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()

    # test 64 va 0.001
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for _ in range(15):
        for batch in loader:
            xs = movedim(batch["x"], 0, 1)
            y = batch["label"]
            
            optimizer.zero_grad()
            loss = languageid_loss(model(xs), y)
            loss.backward()
            optimizer.step()


"""
##################
### QUESTION 5 ###
##################
"""
def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    # Đưa mô hình vào chế độ training
    model.train()

    # Chia dữ liệu thành từng batch, mỗi batch 64 ảnh
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    # Tạo Adam optimizer để cập nhật trọng số
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Huấn luyện qua toàn bộ dữ liệu 10 lần
    for epoch in range(10):

        # Lấy lần lượt từng batch dữ liệu
        for batch in dataloader:

            # Lấy ảnh đầu vào
            x = batch['x']

            # Lấy nhãn đúng dạng one-hot
            y = batch['label']

            # Xóa gradient từ lần học trước
            optimizer.zero_grad()

            # Đưa ảnh vào CNN để dự đoán
            y_pred = model(x)

            # Tính độ lỗi giữa dự đoán và đáp án thật
            loss = digitconvolution_Loss(y_pred, y)

            # Tính gradient để biết cần điều chỉnh trọng số thế nào
            loss.backward()

            # Cập nhật trọng số của mô hình
            optimizer.step()
