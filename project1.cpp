/*
//references: https://stackoverflow.com/questions/68938377/backpropagation-from-scratch-in-c
// https://stackoverflow.com/questions/2019056/getting-a-simple-neural-network-to-work-from-scratch-in-c
 */

#include <iostream>
#include <fstream>
#include <pthread.h>
#include <vector>
#include <cstdlib>
#include <mutex>
#include <semaphore.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ctime>

using namespace std;


///// Declaration of variables

const int no_of_layers = 3;
const int no_of_neurons = 4;
int learning_ratio = 1;
float weights[no_of_layers][no_of_neurons][no_of_neurons];
float biases[no_of_layers][no_of_neurons];
float gradient_weights[no_of_layers][no_of_neurons][no_of_neurons];
float gradient_biases[no_of_layers][no_of_neurons];
mutex mtx;
sem_t semaphore;



/////////////// structure for thread ///////////////

struct ThreadData {
    int layer_no;
    int neuron_no;
    int pipe_fd_weights;   // Pipe for weights
    int pipe_fd_biases;   // Pipe for biases 
};

///// shared memory declaration 

vector<float> shared_memory;


///// func for Neural Network
void neural_networks() {
    // Seed the random number generator
    srand(static_cast<unsigned>(time(nullptr)));

    for (int layer_no = 0; layer_no < no_of_layers; ++layer_no) {
        for (int neuron_no = 0; neuron_no < no_of_neurons; ++neuron_no) {
            for (int prev_neuron_id = 0; prev_neuron_id < no_of_neurons; ++prev_neuron_id) {
                weights[layer_no][neuron_no][prev_neuron_id] = static_cast<float>(rand()) / RAND_MAX;
            }
            biases[layer_no][neuron_no] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}
float forward_propogation(int layer_no, int neuron_no) {
    float input = 0.0;


///// lock

    unique_lock<mutex> lock(mtx);
    for (int prev_neuron_id = 0; prev_neuron_id < no_of_neurons; ++prev_neuron_id) {
        input += weights[layer_no][neuron_no][prev_neuron_id];
    }
    input += biases[layer_no][neuron_no];

    input = max(0.0f, input);

    return input;
}


///// func for backward propogation
//for hidden layer
void backward_propogation(int layer_no, int neuron_no, float error_msg) {
    unique_lock<mutex> lock(mtx);
    biases[layer_no][neuron_no] -= learning_ratio * error_msg;
    for (int prev_neuron_id = 0; prev_neuron_id < no_of_neurons; ++prev_neuron_id) {
        weights[layer_no][neuron_no][prev_neuron_id] -= learning_ratio * error_msg;
    }
}



/////////////// File Handling ///////////////


///// write to file func

void write_to_file_NN(const char* filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    for (int layer_no = 0; layer_no < no_of_layers; ++layer_no) {
        for (int neuron_no = 0; neuron_no < no_of_neurons; ++neuron_no) {
            for (int prev_neuron_id = 0; prev_neuron_id < no_of_neurons; ++prev_neuron_id) {
                file >> weights[layer_no][neuron_no][prev_neuron_id];
            }
            file >> biases[layer_no][neuron_no];
        }
    }

    file.close();
}


///// saving to file func

void save_to_file_NN(const char* filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    for (int layer_no = 0; layer_no < no_of_layers; ++layer_no) {
        for (int neuron_no = 0; neuron_no < no_of_neurons; ++neuron_no) {
            for (int prev_neuron_id = 0; prev_neuron_id < no_of_neurons; ++prev_neuron_id) {
                file << weights[layer_no][neuron_no][prev_neuron_id] << " ";
            }
            file << biases[layer_no][neuron_no] << " ";
        }
        file << endl;
    }

    file.close();
}


/////////////// calculate Gradient ///////////////

void gradient(int layer_no) {
    
///// Calculate the gradient for weights

    for (int i = 0; i < no_of_neurons; ++i) {
        for (int j = 0; j < no_of_neurons; ++j) {
            gradient_weights[layer_no][i][j] = 0.0;
            for (int k = 0; k < no_of_neurons; ++k) {
                gradient_weights[layer_no][i][j] += shared_memory[k] * weights[layer_no][i][k];
            }
        }
    }

///// Calculate the gradient for biases

    for (int i = 0; i < no_of_neurons; ++i) {
        gradient_biases[layer_no][i] = shared_memory[i];
    }
}

void save_to_file_gradient(const char* filename) {
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

///// Save the gradient for weights
    
    for (int layer_no = 0; layer_no < no_of_layers; ++layer_no) {
        for (int i = 0; i < no_of_neurons; ++i) {
            for (int j = 0; j < no_of_neurons; ++j) {
                file << gradient_weights[layer_no][i][j] << " ";
            }
        }
    }

///// Save the gradient for biases

    for (int layer_no = 0; layer_no < no_of_layers; ++layer_no) {
        for (int i = 0; i < no_of_neurons; ++i) {
            file << gradient_biases[layer_no][i] << " ";
        }
    }

    file.close();
}

/////////////// updating Weights and Biases ///////////////

void update_weights_biases(int layer_no) {
    
////// Update weights

    for (int i = 0; i < no_of_neurons; ++i) {
        for (int j = 0; j < no_of_neurons; ++j) {
            weights[layer_no][i][j] -= learning_ratio * gradient_weights[layer_no][i][j];
        }
    }

///// Update biases

    for (int i = 0; i < no_of_neurons; ++i) {
        biases[layer_no][i] -= learning_ratio * gradient_biases[layer_no][i];
    }
}

//////////////// Process Synchronization ///////////////
/////////////// Threads for Neurons ///////////////

void* threads_for_neurons(void* arg) {
    ThreadData* data = (ThreadData*)arg;

///// Wait for semaphore

    sem_wait(&semaphore); 
    unique_lock<mutex> lock(mtx);
    cout << "Layer " << data->layer_no << ", Neuron " << data->neuron_no << " is processing." << endl;

///// Receive input from the shared data

    float input = shared_memory[data->neuron_no];
    lock.unlock();

///// Process the input forward

    input = forward_propogation(data->layer_no, data->neuron_no);

///// Update the shared data
    
    lock.lock();
    shared_memory[data->neuron_no] = input;
    lock.unlock();

///// Perform backward propogation

    backward_propogation(data->layer_no, data->neuron_no, input);

///// Calculate the gradient

    gradient(data->layer_no);

///// Write weights and biases to pipes

    write(data->pipe_fd_weights, weights[data->layer_no][data->neuron_no], sizeof(float) * no_of_neurons);
    write(data->pipe_fd_biases, &biases[data->layer_no][data->neuron_no], sizeof(float));

///// Signal semaphore

    sem_post(&semaphore);
    pthread_exit(NULL);
}

/////////////// creation of Processes for Layers ///////////////
 
void* layers_processes(void* arg) {
    ThreadData* data = (ThreadData*)arg;

///// Create pipes for weights and biases

    int pipe_fd_weights[2];
    int pipe_fd_biases[2];
    if (pipe(pipe_fd_weights) == -1 || pipe(pipe_fd_biases) == -1) {
        cerr << "Pipe creation failed." << endl;
        exit(EXIT_FAILURE);
    }

///// Fork
///// Child process

    pid_t pid = fork();

    if (pid == -1) {
        cerr << "Fork failed." << endl;
        exit(EXIT_FAILURE);
        
    } 
    else if (pid == 0) {

        close(pipe_fd_weights[1]);
        close(pipe_fd_biases[1]);

///// Pass the read ends of the pipes to threads_for_neurons
/////  batch based
 
        ThreadData* neuron_data = new ThreadData;
        neuron_data->layer_no = data->layer_no + 1;
        neuron_data->neuron_no = 0;
        neuron_data->pipe_fd_weights = pipe_fd_weights[0];
        neuron_data->pipe_fd_biases = pipe_fd_biases[0];
        threads_for_neurons(neuron_data);

        close(pipe_fd_weights[0]);
        close(pipe_fd_biases[0]);
        _exit(EXIT_SUCCESS);
    }
    else {
    
        close(pipe_fd_weights[0]);
        close(pipe_fd_biases[0]);

///// Load initial neural network parameters from a file

        write_to_file_NN("Initial_Data_of_Neural_Network.txt");

///// random values in shared memory

        for (int i = 0; i < no_of_neurons; ++i) {
            shared_memory.push_back(static_cast<float>(rand()) / RAND_MAX);
        }

///// Writing initial shared data to the child process

        write(pipe_fd_weights[1], weights[data->layer_no], sizeof(float) * no_of_neurons * no_of_neurons);
        write(pipe_fd_biases[1], biases[data->layer_no], sizeof(float) * no_of_neurons);

///// Create neuron threads

        pthread_t neuron_threads[no_of_neurons];
        for (int i = 0; i < no_of_neurons; ++i) {
            ThreadData* neuron_data = new ThreadData;
            neuron_data->layer_no = data->layer_no + 1;
            neuron_data->neuron_no = i;
            neuron_data->pipe_fd_weights = pipe_fd_weights[1];
            neuron_data->pipe_fd_biases = pipe_fd_biases[1];
            pthread_create(&neuron_threads[i], NULL, threads_for_neurons, neuron_data);
        }

///// Wait for all neuron threads to finish

        for (int i = 0; i < no_of_neurons; ++i) {
            pthread_join(neuron_threads[i], NULL);
        }

///// Read the updated weights and biases from the child process

        read(pipe_fd_weights[1], weights[data->layer_no], sizeof(float) * no_of_neurons * no_of_neurons);
        read(pipe_fd_biases[1], biases[data->layer_no], sizeof(float) * no_of_neurons);

///// Update weights and biases

        update_weights_biases(data->layer_no);

///// Save the trained neural network parameters to a file

        save_to_file_NN("Final_Neural_Network.txt");


        close(pipe_fd_weights[1]);
        close(pipe_fd_biases[1]);

///// Wait for the child process to finish

        waitpid(pid, NULL, 0);

///// Printing the initial weights and biases of NN
        
        cout << "Initial Weights and Biases:" << endl;
        for (int layer_no = 0; layer_no < no_of_layers; ++layer_no) {
            for (int neuron_no = 0; neuron_no < no_of_neurons; ++neuron_no) {
                cout << "Layer " << layer_no << ", Neuron " << neuron_no << ": " << endl;
                cout << "Weights: ";
                for (int prev_neuron_id = 0; prev_neuron_id < no_of_neurons; ++prev_neuron_id) {
                    cout << weights[layer_no][neuron_no][prev_neuron_id] << " ";
                }
                cout << endl;
                cout << "Biases: " << biases[layer_no][neuron_no] << endl;
            }
        }

///// printing the output using weights and biases

        cout << "\nCalculated Output:" << endl;
        for (int layer_no = 0; layer_no < no_of_layers; ++layer_no) {
            for (int neuron_no = 0; neuron_no < no_of_neurons; ++neuron_no) {
                float output = forward_propogation(layer_no, neuron_no);
                cout << "Layer " << layer_no << ", Neuron " << neuron_no << ": " << output << endl;
            }
        }


    }

    return 0;
}


/////////////// Main Function ///////////////

int main() {

///// Neural Network func call
    
    neural_networks();

///// Save initial neural network data to a file

    save_to_file_NN("Initial_Data_of_Neural_Network.txt");
    ThreadData data{0, 0};

///// Initialize semaphore

    sem_init(&semaphore, 0, 1);

///// Fork

    pid_t pid = fork();
    if (pid == -1) {
        cerr << "Fork failed." << endl;
        exit(EXIT_FAILURE);
    }
    else if (pid == 0) {
    
        layers_processes(&data);
        _exit(EXIT_SUCCESS);
    }
    else {
    
///// Parent process
        waitpid(pid, NULL, 0);
    }

///// Destroy semaphore

    sem_destroy(&semaphore);

    return 0;
}


