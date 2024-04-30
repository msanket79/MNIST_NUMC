#include<NeuralNet.hpp>

#include<MNIST/readMNIST.hpp>
#include<MNIST/showMNIST.hpp>

#include<numC/npArrayCpu.hpp>
#include<numC/npFunctions.hpp>
#include<numC/npRandom.hpp>

#include<iostream>
#include<iomanip>
#include<string>
#include<random>
#include<opencv2/opencv.hpp>
#include<vector>




// returns two vectors ,1 of imgs and 1 of labels
// for train,val,test respectively
std::pair<std::vector<float*>, std::vector<int*>> prepareDataset() {
    int num_train_imgs, img_size;
    uchar* train_imgs = readMNISTImages("E:/MNIST_NUMC/modules/MNIST/dataset/train-images.idx3-ubyte", num_train_imgs, img_size);
    std::cout<<std::endl<<"[+]Train Images Loaded\n";

    int num_train_labels;
    uchar* train_labels = readMNISTLabels("E:/MNIST_NUMC/modules/MNIST/dataset/train-labels.idx1-ubyte", num_train_labels);
    std::cout<<std::endl<<"[+]Train Labels Loaded\n";


    int random_idx = (rand() % num_train_imgs);
    showMNIST(train_imgs + random_idx * img_size, 28, 28, std::string(std::string("Ex. train img: ") + std::to_string(train_labels[random_idx])));
    auto randIdxs = np::arange<int>(num_train_imgs);
    np::shuffle(randIdxs);

    const int num_val_imgs = 2000;
    
    float* train_imgs_cpu = (float*)malloc(sizeof(float) * (num_train_imgs - num_val_imgs) * img_size);
    int* train_labels_cpu = (int*)malloc(sizeof(int) * (num_train_imgs - num_val_imgs));
    float* val_imgs_cpu = (float*)malloc(sizeof(float) * (num_val_imgs)*img_size);
    int* val_labels_cpu = (int*)malloc(sizeof(int) * (num_val_imgs));

    for (int i = 0; i < num_val_imgs; i++) {
        int idx = randIdxs.mat[i];
        for (int img_idx = 0; img_idx < img_size; ++img_idx) {
            val_imgs_cpu[i * img_size + img_idx] = train_imgs[idx * img_size + img_idx];
            val_labels_cpu[i] = train_labels[idx];
        }
    }
    for (int i = num_val_imgs; i < num_train_imgs; i++) {
        int idx = randIdxs.mat[i];
        for (int img_idx = 0; img_idx < img_size; ++img_idx) {
            train_imgs_cpu[(i - num_val_imgs) * img_size + img_idx] = train_imgs[idx * img_size + img_idx];
            train_labels_cpu[i - num_val_imgs] = train_labels[idx];
        }
    };
    delete[] train_imgs;
    delete[] train_labels;

    std::cout<<std::endl<<"[+]Val Images Loaded\n";
    std::cout<<std::endl<<"[+]Val Labels Loaded\n";


    int num_test_imgs;
    uchar* test_imgs = readMNISTImages("E:/MNIST_NUMC/modules/MNIST/dataset/t10k-images.idx3-ubyte", num_test_imgs, img_size);

    std::cout<<std::endl<<"[+]Val Images Loaded\n";


    int num_test_labels;
    uchar* test_labels = readMNISTLabels("E:/MNIST_NUMC/modules/MNIST/dataset/t10k-labels.idx1-ubyte", num_test_labels);

    std::cout<<std::endl<<"[+]Val Labels Loaded\n";


    // displaying random image out of train set.
    random_idx = (rand() % num_test_imgs);
    // since it is a 2d array, need to skip 784 (img_size) elements per row.
    showMNIST(test_imgs + random_idx * img_size, 28, 28, std::string(std::string("Test img: ") + std::to_string(test_labels[random_idx])));

    float* test_imgs_cpu = (float*)malloc(num_test_imgs * img_size * sizeof(float));
    int* test_labels_cpu = (int*)malloc(num_test_imgs * sizeof(int));

    for (int i = 0; i < num_test_imgs; ++i)
    {

        for (int img_idx = 0; img_idx < img_size; ++img_idx)
            test_imgs_cpu[i * img_size + img_idx] = test_imgs[i * img_size + img_idx];

        test_labels_cpu[i] = test_labels[i];
    }
    delete[]test_imgs;
    delete[]test_labels;
    std::cout << "\n[+] Done." << std::endl;

 
    return {{ train_imgs_cpu, val_imgs_cpu, test_imgs_cpu },{ train_labels_cpu, val_labels_cpu, test_labels_cpu }};
};




NeuralNet trainModel(float*x_train,int*y_train,int train_size,float*x_val,int*y_val,int val_size,float*x_test,int*y_test,int test_size,int img_size){
    const int batch=512;
    const int num_epochs=20;
    NeuralNet best_model;
    NeuralNet model(0.7315);

    int num_iterations=static_cast<int>(std::ceil(static_cast<double>(train_size) / batch));

    auto x_train_cpu=np::ArrayCpu<float>(x_train,train_size,img_size);

    auto y_train_cpu=np::ArrayCpu<int>(y_train,train_size,1);
    std::cout << "[+] Done." << std::endl;

    std::cout << std::endl;

    auto x_train_batches=np::array_split(x_train_cpu,num_iterations,0);
    auto y_train_batches=np::array_split(y_train_cpu,num_iterations,0);


    std::cout << std::endl
              << "[.] Validation set generated" << std::endl;
    auto x_val_gpu = np::ArrayCpu<float>(x_val, val_size, img_size);

    auto y_val_gpu = np::ArrayCpu<int>(y_val, val_size, 1 );
    std::cout << "[+] Done." << std::endl;
    x_train_cpu = np::ArrayCpu<float>(1, 1);
    y_train_cpu = np::ArrayCpu<int>(1, 1);
    // for storing the best model

        std::cout << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cout << "------------Beginning Network Training--------------" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
  
    float best_val_acc = 0, best_train_acc = 0;
    float val_acc = 0, train_acc = 0;


        auto st = clock();
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {

        for (int iter = 0; iter < num_iterations; ++iter)
        {

            // converting to train mode
            model.train();

            //forward pass
            auto outNloss = model.forward(x_train_batches[iter], y_train_batches[iter]);

            model.adamStep();


            if ((iter + 1) % 100 == 0)
            {
                auto predicted_gpu = outNloss.first.argmax(1);

                train_acc = static_cast<float>(((predicted_gpu == y_train_batches[iter]).sum()).mat[0]) / y_train_batches[iter].rows;
                // evaluating on validation set
                model.eval();
                auto y_pred_gpu = model(x_val_gpu);
                predicted_gpu = y_pred_gpu.argmax(1);

                val_acc = static_cast<float>(((predicted_gpu == y_val_gpu).sum()).mat[0]) / val_size;

                std::cout << "Epoch: " << epoch + 1 << " iter: " << iter + 1 << " loss: " << outNloss.second << " train_acc: " << train_acc << " val_acc: " << val_acc << std::endl;
            }
        }
        if ((best_val_acc < val_acc) || (best_val_acc == val_acc && best_train_acc < train_acc))
        {
            best_val_acc = val_acc;
            best_train_acc = train_acc;
            best_model = model;
            std::cout << std::endl
                      << "##################### NEW BEST FOUND! ###########################" << std::endl;
            std::cout << "##################### VAL ACC: " << std::fixed << std::setprecision(3) << best_val_acc << "                           ##" << std::endl;
            std::cout << "##################### TRAIN ACC: " << std::fixed << std::setprecision(3) << train_acc << "                         ##" << std::endl;
            std::cout << "#################################################################" << std::endl
                      << std::endl;
        }
    }

        auto end = clock();
    std::cout << std::endl
              << "TOTAL TIME: " << static_cast<double>(end - st) / CLOCKS_PER_SEC << " s" << std::endl;

    std::cout << std::endl
              << "[+] Model Training Done." << std::endl;

    std::cout << std::endl
              << "[.] Loading test set on GPU" << std::endl;
    auto x_test_gpu = np::ArrayCpu<float>(x_test, test_size, img_size);

    auto y_test_gpu = np::ArrayCpu<int>(y_test, test_size, 1);
    std::cout << "[+] Done." << std::endl;

    std::cout << std::endl
              << "[.] Performing analysis on test set" << std::endl;
    best_model.eval();
    auto y_pred_gpu = best_model(x_test_gpu);
    auto predicted_gpu = y_pred_gpu.argmax(1);

    auto test_acc = static_cast<float>(((predicted_gpu == y_test_gpu).sum()).get(0)) / y_pred_gpu.rows;
    std::cout << "[+] Done." << std::endl;

    std::cout << std::endl
              << "####### Final model stats: #######" << std::endl;
    std::cout << "####### TRAIN ACC: " << std::fixed << std::setprecision(3) << best_train_acc << "        ##" << std::endl;
    std::cout << "####### VAL ACC: " << std::fixed << std::setprecision(3) << best_val_acc << "          ##" << std::endl;
    std::cout << "####### TEST ACC: " << std::fixed << std::setprecision(3) << test_acc << "         ##" << std::endl;
    std::cout << "##################################" << std::endl;

    return best_model;



    
}
template<typename T>
void test(T* A, T* B, int rows, int cols) {
    int flag = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (A[i] != B[i]) {
            flag = 1;
            break;
        }
    }
    if (flag == 0) std::cout << "test passed" << std::endl;
    else std::cout << "test failed" << std::endl;
}
template<typename T>
void dot(T* A, T* B, T* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
      }
}

void dot_test(int m, int n, int k) {
    auto A = np::Random::rand<int>(m, k, 1, 10);
    auto B = np::Random::rand<int>(k, n, 1, 10);
    auto C = np::ArrayCpu<int>(m, n, 0);
    auto C1 = A.dot(B);
 
    dot<int>(A.mat, B.mat, C.mat, m, n, k);
    test(C.mat, C1.mat, m, n);

}
void dotT_test(int m, int n, int k) {
    auto A = np::Random::rand<int>(m, k, 1, 10);
    auto B = np::Random::rand<int>(n, k, 1, 10);
    auto C = np::ArrayCpu<int>(m, n, 0);
    auto C1 = A.dotT(B);
    B = B.T();
    dot<int>(A.mat, B.mat, C.mat, m, n, k);
    test(C.mat, C1.mat, m, n);

}
void Tdot_test(int m, int n, int k) {
    auto A = np::Random::rand<int>(k, m, 1, 10);
    auto B = np::Random::rand<int>(k, n, 1, 10);
    auto C = np::ArrayCpu<int>(m, n, 0);
    auto C1 = A.Tdot(B);
    A = A.T();
    dot<int>(A.mat, B.mat, C.mat, m, n, k);
    test(C.mat, C1.mat, m, n);

}
int main(){

    //  NeuralNet N(1);
    // auto A=np::Random::randn<float>(1000,784);
    // auto Y=np::Random::rand<int>(1000,1,0,9);
    
    
    // // //std::cout << Y;
    // ////std::cout<<L(A);
    //   for(int i=0;i<10;i++){
    //       N.eval();
    //       auto C=N.forward(A,Y);
    //       N.adamStep();
    //       std::cout <<C.second;
    //       //std::cout << (C.first.argmax(1)==Y).sum().mat[0]/1000.0;
    //       //std::cout << (C.first.argmax(1) == Y);
    //       //std::cout<<((C.first.argmax(1)==Y).sum())/10000<<std::endl;
    //   }

    // ################### SOFTMAX DEBUG
    auto y = np::Random::randn<float>(9, 10);
    std::cout<<y<<"\n\n";
    int *a = new int[9]{1, 4, 5, 2, 3, 9, 7, 6, 0};
    auto y_actual = np::ArrayCpu<int>(9);
    for(int i = 0; i< 9; ++i)
        y_actual(0, i) = a[i];

    std::cout<<y_actual<<"\n\n";


    std::cout<<"\nLOSS: "<<SoftmaxLoss::computeLossAndGrad(y, y_actual)[0]<<"\nDx: "<<SoftmaxLoss::computeLossAndGrad(y, y_actual)[1];
    
    //  std::cout << std::endl
    //            << "----------------------------------------------------" << std::endl;
    //  std::cout << "----------------STARTING DATA FETCH-----------------" << std::endl;
    //  std::cout << "----------------------------------------------------" << std::endl;
    //   auto imgsNlabels = prepareDataset();

    //  std::cout << std::endl
    //            << "----------------------------------------------------" << std::endl;
    //  std::cout << "-------------IMAGES AND LABELS FETCHED--------------" << std::endl;
    //  std::cout << "----------------------------------------------------" << std::endl;

    //  std::cout << std::endl
    //            << "----------------------------------------------------" << std::endl;
    //  std::cout << "-------------Beginning GPU execution----------------" << std::endl;
    //  std::cout << "----------------------------------------------------" << std::endl;
    //   NeuralNet nn = trainModel(imgsNlabels.first[0], imgsNlabels.second[0], 58000, imgsNlabels.first[1], imgsNlabels.second[1], 2000, imgsNlabels.first[2], imgsNlabels.second[2], 10000, 784);
    //   nn.eval();
}









