#include "bert.h"

Bert::Bert(int party, int port, string address){
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

    cout << "Setup Linear" << endl;
    this->lin = Linear(party, io);
    cout << "Setup NonLinear" << endl;
    this->nl = NonLinear(party, address, port+1);
    cout << "Setup NonLinear done" << endl;
}

Bert::~Bert() {
    
}

void Bert::run() {

}