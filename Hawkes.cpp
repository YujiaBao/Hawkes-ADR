/**********************************************************
 * Author: Yujia Bao
 * Email : yujia@csail.mit.edu
 * Last modified : 2017-08-22 19:58
 * Filename  : Hawkes.cpp
 * Copyright(c) 2017, Yujia Bao All Rights Reserved.
 * Description   :  Code for MLHC 2017: Hawkes Process Modeling of Adverse Drug Reactions with Longitudinal Observational Data
 * *******************************************************/

#include<iostream>
#include<fstream>
#include<algorithm>
#include<vector>
#include<list>
#include<sstream>
#include<iomanip>
#include<string>
#include<cmath>
#include<cstdlib>
#include<limits>
#include<thread>
#include<random>
#include<sstream>
#include<unistd.h>

using namespace std;

#define HOSPITALIZATION_INDEX 7
#define SKIP_HOSPITALIZATION false // hospitalization is not in the 53 pairs. Skip it can improve running time.

#define DRUG_ONLY true // only use drug to predict outcomes, otherwise use drug + outcome to predict outcome

#define numOfVariables 20 // num. of outcome + num. of drug
#define numOfOutcomes 10 // num. of outcome
#define numOfDrugs 10 // num. of drugs
// assume outcome index: 1 to numOfOutcomes
// assume drug index: numOfOutcomes+1 to numOfVariables

#define THREADS  24 // change it if you are running the code on your laptop

class dataPackage {
public:
    // hyper-parameter settings
    vector<int> start;
    vector<int> end;
    vector<double> value;

    // data statistics
    vector<vector<pair<int, int>>> data;
    vector<vector<int>> counts;
    vector<vector<vector<double>>> cumInfo;
    vector<vector<vector<vector<double>>>> wholeCumInfo;
    vector<vector<vector<double>>> cumSum;
    vector<vector<double>> wholeDuration;
    double totalDataTime;
    double totalDataPatients;

    // model parameters
    vector<vector<double>> baseline;
    vector<vector<vector<double>>> weights;
    vector<vector<vector<double>>> gradWeights;

    // output results
    double loss;
    bool status;
    int numOfKernels;

    // initialization
    //
    dataPackage (vector<vector<pair<int, int>>> &d, int i, vector<int> &kernelStart, vector<int> &kernelEnd, vector<double> &kernelValue) {
        int numOfPatients = d.size();
        data.insert(data.begin(), d.begin()+numOfPatients/THREADS*i, d.begin()+numOfPatients/THREADS*(i+1));
        numOfPatients = data.size();
        numOfKernels = kernelValue.size(); 

        counts = vector<vector<int>> (numOfPatients, vector<int>(numOfOutcomes, 0));

        if (DRUG_ONLY)
            cumInfo = vector<vector<vector<double>>> (numOfOutcomes,  vector<vector<double>>(numOfDrugs, vector<double> (numOfKernels, 0)));
        else
            cumInfo = vector<vector<vector<double>>> (numOfOutcomes,  vector<vector<double>>(numOfVariables, vector<double> (numOfKernels, 0)));

        wholeCumInfo = vector<vector<vector<vector<double>>>> (numOfPatients, vector<vector<vector<double>>>());

        wholeDuration = vector<vector<double>> (numOfPatients, vector<double>());

        baseline = vector<vector<double>>(data.size(), vector<double>(numOfOutcomes,0));

        cumSum = vector<vector<vector<double>>>(data.size(),vector<vector<double>> (numOfOutcomes, vector<double>()));
        start = kernelStart;
        end = kernelEnd;
        value = kernelValue;
    }

    // reload weight information and clear the gradients
    void load(vector<vector<vector<double>>> &w) {
        loss = 0;
        weights = w;
        gradWeights = w;
        for (int i = 0; i < gradWeights.size(); i++) {
            for (int j = 0; j < gradWeights[0].size(); j++)
                for (int k = 0; k < gradWeights[0][0].size(); k++)
                    gradWeights[i][j][k] = 0;
        }
    }
};

vector<vector<pair<int, int>>> readLines (string filePath) {
    ifstream ifile(filePath);
    vector<vector<pair<int, int>>> data;
    string line;
    int time, event;
    while (getline(ifile, line)) {
        stringstream ss;
        ss << line;
        vector<pair<int, int>> curPatient;
        while (ss >> time) {
            ss >> event;
            curPatient.push_back(make_pair(time, event-1));
        }
        data.push_back(curPatient);
    }
    ifile.close();
    return data;
}

void getCumInfo (dataPackage *info) {
    auto &data = info->data;
    auto &counts = info->counts;
    auto &start = info->start;
    auto &end = info->end;
    auto &value = info->value;
    auto &cumInfo = info->cumInfo;
    auto &totalDataTime = info->totalDataTime;
    auto &wholeCumInfo = info->wholeCumInfo;
    auto &cumSum = info->cumSum;
    auto &wholeDuration = info->wholeDuration;
    int numOfPatients = data.size();
    int numOfKernels = info->numOfKernels;

    for (int p = 0; p < numOfPatients; p++)
         totalDataTime += data[p].back().first - data[p].front().first;

    // compute the all cumulative information, all subintervals
    for (int p = 0; p < numOfPatients; p++) {
        // get the counts for the outcomes
        for (auto event : data[p])
            if (event.second < numOfOutcomes)
                counts[p][event.second]++;

        // initialize the whole interval
        vector<int> intervalStart, intervalEnd;
        intervalStart.push_back(data[p].front().first);
        intervalEnd.push_back(data[p].back().first);

        if (DRUG_ONLY)
            wholeCumInfo[p].push_back(vector<vector<double>>(numOfDrugs, vector<double> (numOfKernels, 0.0))); // only use drug info
        else
            wholeCumInfo[p].push_back(vector<vector<double>>(numOfVariables, vector<double> (numOfKernels, 0.0))); // use drug info + event info

        int maxTime = data[p].back().first;

        for (auto event : data[p]) {
            if (event.first == maxTime) // last event
                break;
            int eventLabel;
            if (DRUG_ONLY) {
                if (event.second < numOfOutcomes) // skip, this is outcome
                    continue;
                else
                    eventLabel = event.second-numOfOutcomes;
            } else {
                eventLabel = event.second;
            }

            for (int k = 0; k < numOfKernels; k++) {
                int effectStart = min(event.first+start[k], maxTime);
                int effectEnd = min(event.first+end[k], maxTime);
                if (effectStart == maxTime) // at the end of the trajectory
                    continue;

                int i = 0;
                for (; i < intervalEnd.size(); i++)
                    if (effectStart < intervalEnd[i])
                        break;

                if (intervalStart[i] < effectStart) {
                    // split intervals if needed
                    intervalStart.insert(intervalStart.begin()+i, intervalStart[i]);
                    intervalEnd.insert(intervalEnd.begin()+i, effectStart);
                    intervalStart[i+1] = intervalEnd[i];
                    wholeCumInfo[p].insert(wholeCumInfo[p].begin()+i, wholeCumInfo[p][i]);
                    i++;
                }
                // add value to the origin intervals, if exist
                //
                for (;i < intervalEnd.size(); i++) {
                    if (intervalEnd[i] < effectEnd)
                        wholeCumInfo[p][i][eventLabel][k] += value[k];
                    else if (intervalEnd[i] == effectEnd) {
                        wholeCumInfo[p][i][eventLabel][k] += value[k];
                        i++;
                        break;
                    } else {
                        // split
                        intervalStart.insert(intervalStart.begin()+i, intervalStart[i]);
                        intervalEnd.insert(intervalEnd.begin()+i, effectEnd);
                        intervalStart[i+1] = intervalEnd[i];
                        wholeCumInfo[p].insert(wholeCumInfo[p].begin()+i, wholeCumInfo[p][i]);
                        wholeCumInfo[p][i][eventLabel][k] += value[k];
                        i++;
                        break;
                    }
                }
            }
        }
        for (int i = 0; i < intervalStart.size(); i++)
            wholeDuration[p].push_back(-intervalStart[i] + intervalEnd[i]);
        for (int cond = 0; cond < numOfOutcomes; cond++)
            cumSum[p][cond] = vector<double>(intervalStart.size(),0);
    }

    // compute the cumulative information for predicting the occurrence of some event
    for (int i = 0; i < numOfOutcomes; i++) {
        for (int p = 0; p < numOfPatients; p++) {
            for (int j = 0; j < data[p].size(); j++) {
                if (data[p][j].second == i) {
                    if (DRUG_ONLY) {
                        int currentTime = data[p][j].first;
                        for (int l = 0; l < data[p].size() && data[p][l].first <= currentTime; l++) {
                            if (l == j)
                                continue;
                            if (data[p][l].second < numOfOutcomes)
                                continue;
                            for (int k = 0; k < numOfKernels; k++)
                                if (currentTime >= data[p][l].first+start[k] && currentTime < data[p][l].first+end[k])
                                    cumInfo[i][data[p][l].second-numOfOutcomes][k] += value[k];
                        }
                    } else {
                        int currentTime = data[p][j].first;
                        for (int l = 0; l < data[p].size() & data[p][l].first <= currentTime; l++) {
                            if (l == j)
                                continue;
                            for (int k = 0; k < numOfKernels; k++)
                                if (currentTime >= data[p][l].first+start[k] && currentTime < data[p][l].first+end[k])
                                    cumInfo[i][data[p][l].second][k] += value[k];
                        }
                    }
                }
            }
        }
    }
}

void getGradients (dataPackage *info) {
    auto &counts = info->counts;
    auto &cumInfo = info->cumInfo;
    auto &wholeCumInfo = info->wholeCumInfo;
    auto &wholeDuration = info->wholeDuration;
    auto &cumSum = info->cumSum;
    auto &data = info->data;
    auto &weights = info->weights;
    auto &gradWeights = info->gradWeights;
    auto &baseline = info->baseline;
    auto &loss = info->loss;
    int numOfPatients = data.size();
    int predictionVariables = gradWeights[0].size();
    int numOfKernels = weights[0][0].size();

    for (int i = 0; i < numOfOutcomes; i++) {
        if (SKIP_HOSPITALIZATION && i == (HOSPITALIZATION_INDEX-1)) // skip predicting hospitalization
            continue;
        for (int j = 0; j < predictionVariables; j++)
            for (int k = 0; k < numOfKernels; k++) {
                gradWeights[i][j][k] -= cumInfo[i][j][k];
                loss -= weights[i][j][k] * cumInfo[i][j][k];
            }
    }

    for (int p = 0; p < numOfPatients; p++) {
        for (int i = 0; i < numOfOutcomes; i++) {
            if (SKIP_HOSPITALIZATION && i == (HOSPITALIZATION_INDEX-1)) // skip predicting hospitalization
                continue;

            if (counts[p][i] == 0) // self-controlled designed (skip if the patient never has this disease
                continue;

            double expBaseline = exp(baseline[p][i]);
            for (int ii = 0; ii < wholeDuration[p].size(); ii++) {
                double intensity = cumSum[p][i][ii] * wholeDuration[p][ii] * expBaseline;
                loss += intensity;
                for (int j = 0; j < predictionVariables; j++)
                    for (int k = 0; k < numOfKernels; k++)
                        gradWeights[i][j][k] += intensity * wholeCumInfo[p][ii][j][k];
            }
        }
    }
}

void updateBaseline (dataPackage *info) {
    auto &counts = info->counts;
    auto &cumInfo = info->cumInfo;
    auto &wholeCumInfo = info->wholeCumInfo;
    auto &wholeDuration = info->wholeDuration;
    auto &totalDataTime = info->totalDataTime;
    auto &totalDataPatients = info->totalDataPatients;
    auto &cumSum = info->cumSum;
    auto &data = info->data;
    auto &weights = info->weights;
    auto &baseline = info->baseline;
    auto &loss = info->loss;
    int numOfPatients = data.size();
    int predictionVariables = weights[0].size();
    int numOfKernels = weights[0][0].size();
    info->status = true;

    // update baseline according to maximum likelihood estimate and the regularization
    for (int p = 0; p < numOfPatients; p++) {
        for (int i = 0; i < numOfOutcomes; i++) {
            if (SKIP_HOSPITALIZATION && i == (HOSPITALIZATION_INDEX-1)) // skip predicting hospitalization
                continue;
            if (counts[p][i] == 0) // self-controlled design
                continue;

            // calculate integral
            double integral = 0, tmp;
            for (int ii = 0; ii < wholeDuration[p].size(); ii++) {
                tmp = 0;
                for (int j = 0; j < predictionVariables; j++)
                    for (int k = 0; k < numOfKernels; k++)
                        tmp += weights[i][j][k] * wholeCumInfo[p][ii][j][k];
                tmp = exp(tmp);
                cumSum[p][i][ii] = tmp;
                integral += tmp * wholeDuration[p][ii];
            }

            if (std::isinf(integral) || std::isnan(integral)) {
                cerr << "Invalid integral value. Need to shrink step size for FISTA." << endl;
                exit(1);
            }

            baseline[p][i] = log(counts[p][i]/integral);
            loss -= counts[p][i] * baseline[p][i];
        }
    }
}

bool getWholeGradients (vector<vector<vector<double>>> &weights, vector<vector<vector<double>>> &gradWeights,
        dataPackage* pac[THREADS], double &loss, double &norm, double &lambda1) {
    int numOfKernels = weights[0][0].size();
    thread threadList[THREADS];
    //cout << "Load" << endl;
    for (int i = 0; i < THREADS; i++)
        pac[i]->load(weights);

    // update baseline
    for (int i = 0; i < THREADS; i++)
        threadList[i] = thread(updateBaseline, pac[i]);
    for (int i = 0; i < THREADS; i++)
        threadList[i].join();
    for (int i = 0; i < THREADS; i++)
        if (pac[i]->status == false)
            return false;

    // get gradients for the weights
    for (int i = 0; i < THREADS; i++)
        threadList[i] = thread(getGradients, pac[i]);
    for (int i = 0; i < THREADS; i++)
        threadList[i].join();

    //cout << "get merge gradients" << endl;
    norm = 0;
    int predictionVariables = gradWeights[0].size();
    for (int i = 0; i < numOfOutcomes; i++) {
        for (int j = 0; j < predictionVariables; j++) {
            for (int k = 0; k < numOfKernels; k++) {
                gradWeights[i][j][k] = 0.0;
                for (int t = 0; t < THREADS; t++)
                    gradWeights[i][j][k] += pac[t]->gradWeights[i][j][k];
                if (weights[i][j][k] > 1e-10)
                    norm = max(abs(gradWeights[i][j][k]+ lambda1), norm);
                else if (weights[i][j][k] < -1e-10)
                    norm = max(abs(gradWeights[i][j][k]-lambda1), norm);
                else if (gradWeights[i][j][k] > lambda1)
                    norm = max(abs(gradWeights[i][j][k]-lambda1), norm);
                else if (gradWeights[i][j][k] < -lambda1)
                    norm = max(abs(gradWeights[i][j][k]+lambda1), norm);
            }
        }
    }

    loss = 0;
    for (int t = 0; t < THREADS; t++)
        loss += pac[t]->loss;
    return true;
}

void getLoss (dataPackage *info) {
    auto &counts = info->counts;
    auto &cumInfo = info->cumInfo;
    auto &wholeCumInfo = info->wholeCumInfo;
    auto &wholeDuration = info->wholeDuration;
    auto &data = info->data;
    auto &weights = info->weights;
    auto &baseline = info->baseline;
    auto &cumSum = info->cumSum;
    auto &loss = info->loss;
    int predictionVariables = weights[0].size();
    int numOfPatients = data.size();
    int numOfKernels = weights[0][0].size();

    for (int i = 0; i < numOfOutcomes; i++) {
        if (SKIP_HOSPITALIZATION && i == (HOSPITALIZATION_INDEX-1)) // skip predicting hospitalization
            continue;

        for (int j = 0; j < predictionVariables; j++)
            for (int k = 0; k < numOfKernels; k++)
                loss -= weights[i][j][k] * cumInfo[i][j][k];
    }

    for (int p = 0; p < numOfPatients; p++) {
        for (int i = 0; i < numOfOutcomes; i++) {
            if (SKIP_HOSPITALIZATION && i == (HOSPITALIZATION_INDEX-1)) // skip predicting hospitalization
                continue;
            if (counts[p][i] == 0)
                continue;

            double expBaseline = exp(baseline[p][i]);
            for (int ii = 0; ii < wholeDuration[p].size(); ii++)
                loss += cumSum[p][i][ii] * wholeDuration[p][ii] * expBaseline;
        }
    }
}

double getWholeLoss (vector<vector<vector<double>>> &weights, dataPackage* pac[THREADS]) {
    thread threadList[THREADS];
    //cout << "Load" << endl;
    for (int i = 0; i < THREADS; i++)
        pac[i]->load(weights);

    //cout << "Update" << endl;
    for (int i = 0; i < THREADS; i++)
        threadList[i] = thread(updateBaseline, pac[i]);
    for (int i = 0; i < THREADS; i++)
        threadList[i].join();

    //cout << "Calculate" << endl;
    for (int i = 0; i < THREADS; i++)
        threadList[i] = thread(getLoss, pac[i]);
    for (int i = 0; i < THREADS; i++)
        threadList[i].join();

    //cout << "Merge" << endl;
    double loss = 0;
    for (int t = 0; t < THREADS; t++)
        loss += pac[t]->loss;

    return loss;
}

void preprocess (string &filePath, dataPackage* pac[THREADS], vector<int> start, vector<int> end, vector<double> value) {
    auto data = readLines(filePath);
    cout << data.size() << endl;
    thread threadList[THREADS];
    for (int i = 0; i < THREADS; i++) {
        pac[i] = new dataPackage(data, i, start, end, value);
    }

    for (int i = 0; i < THREADS; i++)
        threadList[i] = thread(getCumInfo, pac[i]);
    double totalDataTime = 0, totalDataPatients = 0;
    for (int i = 0; i < THREADS; i++) {
        threadList[i].join();
        totalDataTime += pac[i]->totalDataTime;
        totalDataPatients += pac[i]->data.size();
    }
    for (int i = 0; i < THREADS; i++) {
        pac[i]->totalDataTime = totalDataTime;
        pac[i]->totalDataPatients = totalDataPatients;
    }
}

double lassoLoss (double loss, vector<vector<vector<double>>> &weights, double lambda) {
    for (int i = 0; i < weights.size(); i++)
        for (int j = 0; j < weights[0].size(); j++)
            for (int k = 0; k < weights[0][0].size(); k++)
                loss += abs(weights[i][j][k]) * lambda;
    return loss;
}

void updateParameters (vector<vector<vector<double>>> &weights, vector<vector<vector<double>>> &gradWeights, vector<vector<vector<double>>> &yWeights, double t, double &tk, double lambda1) {
    // FISTA
    auto prevWeights = weights;
    double newtk = (1.0+sqrt(1+4.0*tk*tk))/2.0;
    double tmp = (tk-1.0)/newtk;
    tk = newtk;
    for (int i = 0; i < numOfOutcomes; i++) {
        for (int j = 0; j < weights[0].size(); j++) {
            for (int k = 0; k < weights[0][0].size(); k++) {
                weights[i][j][k] = yWeights[i][j][k] - t * gradWeights[i][j][k];
                if (abs(weights[i][j][k]) <= t * lambda1)
                    weights[i][j][k] = 0;
                else if (weights[i][j][k] > t * lambda1)
                    weights[i][j][k] -= t*lambda1;
                else
                    weights[i][j][k] += t*lambda1;
                yWeights[i][j][k] = weights[i][j][k] + tmp * (weights[i][j][k] - prevWeights[i][j][k]);
            }
        }
    }
}

void printStatus (ostream &ofile, int n, double oldLoss, double newLoss, double norm, double trainPatients) {
    ofile << "Iter: ";
    ofile << setw(5) << n;
    ofile << ", training loss: ";
    ofile << fixed << setprecision(13) << newLoss/trainPatients ;
    ofile << ", gradient norm: ";
    ofile << fixed << setprecision(13) << norm/trainPatients;
    ofile << ", Improvement: ";
    ofile << fixed << setprecision(13) << (oldLoss-newLoss)/trainPatients << endl;
}

void printParameters (ofstream &ofile, vector<int> &start, vector<int> &end, vector<vector<vector<double>>> &weights) {
    ofile << "\n\nWeights: " << endl;
    int numOfKernels = weights[0][0].size();
    for (int k = 0; k < numOfKernels; k++) {
        ofile << "Start: " << start[k] << ", End: " << end[k] << endl;
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[0].size(); j++)
                ofile << setprecision(6) << setw(12) << weights[i][j][k] << " ";
            ofile << endl;
        }
        ofile << "\n" << endl;
    }

    if (DRUG_ONLY) {
        ofile << "\nInfluence from drugs to outcomes" << endl;
        ofile << "Row for outcomes, and column for drugs: w_{ij} represents the effect of drug j on outcome i" << endl;
        for (int i = 0; i < numOfOutcomes; i++) {
            double norm = 0;
            vector<double> sum(numOfDrugs, 0);
            for (int j = 0; j < numOfDrugs; j++) {
                for (int k = 0; k < numOfKernels; sum[j]+=weights[i][j][k++]);
                norm += sum[j] * sum[j];
            }
            norm = 1.0/sqrt(norm);
            for (int j = 0; j < numOfDrugs; j++) 
                ofile << setprecision(6) << setw(12) << sum[j]*norm << " ";
            ofile << endl;
        }
    } else {
        ofile << "\nInfluence from drugs to outcomes" << endl;
        ofile << "Row for outcomes, and column for drugs: w_{ij} represents the effect of drug j on outcome i" << endl;
        for (int i = 0; i < numOfOutcomes; i++) {
            double norm = 0;
            vector<double> sum(numOfDrugs, 0);
            for (int j = numOfOutcomes; j < numOfVariables; j++) {
                for (int k = 0; k < numOfKernels; sum[j-numOfOutcomes]+=weights[i][j][k++]);
                norm += sum[j] * sum[j];
            }
            norm = 1.0/sqrt(norm);
            for (int j = 0; j < numOfDrugs; j++) 
                ofile << setprecision(6) << setw(12) << sum[j]*norm << " ";
            ofile << endl;
        }
    }
}

void printInfluence (ofstream &ofile, vector<int> &start, vector<int> &end, vector<vector<vector<double>>> &weights) {
    int numOfKernels = weights[0][0].size();
    if (DRUG_ONLY) {
        ofile << "\nInfluence from drugs to outcomes" << endl;
        ofile << "Row for outcomes, and column for drugs: w_{ij} represents the effect of drug j on outcome i" << endl;
        for (int i = 0; i < numOfOutcomes; i++) {
            double norm = 0;
            vector<double> sum(numOfDrugs, 0);
            for (int j = 0; j < numOfDrugs; j++) {
                for (int k = 0; k < numOfKernels; sum[j]+=weights[i][j][k++]);
                norm += sum[j] * sum[j];
            }
            norm = 1.0/sqrt(norm);
            for (int j = 0; j < numOfDrugs; j++) 
                ofile << setprecision(6) << setw(12) << sum[j]*norm << " ";
            ofile << endl;
        }
    } else {
        ofile << "\nInfluence from drugs to outcomes" << endl;
        ofile << "Row for outcomes, and column for drugs: w_{ij} represents the effect of drug j on outcome i" << endl;
        for (int i = 0; i < numOfOutcomes; i++) {
            double norm = 0;
            vector<double> sum(numOfDrugs, 0);
            for (int j = numOfOutcomes; j < numOfVariables; j++) {
                for (int k = 0; k < numOfKernels; sum[j-numOfOutcomes]+=weights[i][j][k++]);
                norm += sum[j] * sum[j];
            }
            norm = 1.0/sqrt(norm);
            for (int j = 0; j < numOfDrugs; j++) 
                ofile << setprecision(6) << setw(12) << sum[j]*norm << " ";
            ofile << endl;
        }
    }
}

int main(int argc, char *argv[]) {
    // Initialize Random Seed
    default_random_engine gen(1);

    // default configuration
    int W = 500; // the length of the time-at-risk window, it is L in the paper
    int numOfKernels = 4; // num. of kernels
    double lambda1 = 0; // lasso
    string filePath = "Data/syn_example.txt";

    // Parsing the arguments
    int argStatus;
    while ((argStatus = getopt(argc, argv, "w:l:k:f:")) != -1) {
        switch (argStatus) {
            case 'w':
                W = atoi(optarg);
                cout << "L is set to: " << W << endl;
                break;
            case 'l':
                lambda1 = atof(optarg);
                cout << "Lasso parameter set to: " << lambda1 << endl;
                break;
            case 'k':
                numOfKernels = atoi(optarg);
                cout << "Num. of kernels is set to: " << numOfKernels << endl;
                break;
            case 'f':
                filePath = optarg;
                cout << "File path set to: " << filePath << endl;
                break;
			case '?':
				if (optopt == 'w' || optopt == 'l' || optopt == 'k' || optopt == 'f')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
                    fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
				return 1;
            default:
                abort ();
        }
    }
    
    // output file
    ofstream ofile;
    string fileName = to_string(time(0)) + "_";
    if (!DRUG_ONLY)
        fileName += "ALLVAR_";
    fileName += "LASSO_" + to_string(lambda1) + "_";
    fileName += "L" + to_string(W) + "_" + "K" + to_string(numOfKernels);
    ofile.open("Result/" + fileName + ".txt");
    cout << "Output file: \n" << "Result/" + fileName + ".txt\n" << endl;

    // dyadic kernels
    vector<int> start(numOfKernels, 0), end(numOfKernels, 0);
    for (int i = 1; i <= numOfKernels; i++) {
        end[numOfKernels-i] = W;
        start[numOfKernels-i] = end[numOfKernels-i]/2;
        W /= 2;
    }
    vector<double> value(numOfKernels, 0);
    for (int i = 0; i < numOfKernels; i++)
        value[i] = 1.0/(end[i]-start[i]);

    // print kernels:
    ofile << "Kernels: " << endl;
    for (int i = 0; i < numOfKernels; i++) {
        ofile  << "Kernel " << i << ": from " << start[i] << " to " << end[i] << " with value: " << value[i] << endl;
        cout   << "Kernel " << i << ": from " << start[i] << " to " << end[i] << " with value: " << value[i] << endl;
    }
    ofile << "L1: " << lambda1 << endl;
    cout << "L1: " << lambda1 << endl;

    // this is for multithreading
    dataPackage *pac[THREADS];
    auto timeStart = chrono::system_clock::now();
    preprocess(filePath, pac, start, end, value);
    auto timeEnd = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = timeEnd-timeStart;
    ofile << "It took " << elapsed_seconds.count() << " seconds to preprocess the training data" << endl;
    cout << "It took " << elapsed_seconds.count() << " seconds to preprocess the training data" << endl;


    // Initialize Parameters
    vector<vector<vector<double>>> weights;
    if (DRUG_ONLY)
        weights = vector<vector<vector<double>>>(numOfOutcomes, vector<vector<double>> (numOfDrugs, vector<double>(numOfKernels, 0)));
    else
        weights = vector<vector<vector<double>>>(numOfOutcomes, vector<vector<double>> (numOfVariables, vector<double>(numOfKernels, 0)));

    auto yWeights = weights;
    auto gradWeights = weights;
    double eta = 0.005; //learning rate
    double norm; // measure the gradient norm
    double loss = getWholeLoss(weights, pac);
    double trainPatients = pac[0]->totalDataPatients;

    ofile << "Eta: " << eta << endl;
    printStatus(ofile, 0, loss, loss, 0, trainPatients);
    printStatus(cout, 0, loss, loss, 0, trainPatients);

    lambda1 *= pac[0]->totalDataPatients; // This is equivalent to minimizing  avg. patient loss + lambda1 * |w|_1

    double tk = 1; // FISTA's parameter
    for (int n = 1, subCycle = 0; ; n++) {
        double newLoss;
        getWholeGradients(yWeights, gradWeights, pac, newLoss, norm, lambda1);
        updateParameters(weights, gradWeights, yWeights, eta, tk, lambda1);

        printStatus(ofile, n, loss, newLoss, norm, trainPatients);
        printStatus(cout, n, loss, newLoss, norm, trainPatients);

        //if ((loss-newLoss)/trainPatients > 5e-6) {
        if ((loss-newLoss)/trainPatients > 1e-4) {
            // sufficient improvement over the last best
            subCycle = 0;
        } else
            subCycle++;

        if (subCycle >= 20 && norm/trainPatients < 5e-4 ) {
            // terminate learning
            ofile << "End of learning" << endl;
            printInfluence(ofile, start, end, weights);
            break;
        }
        loss = newLoss;
    }

    cout << fileName << endl;
    return 0;
}
