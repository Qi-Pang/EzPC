/*
This is an autogenerated file, generated using the EzPC compiler.
*/

#include "emp-sh2pc/emp-sh2pc.h" 
using namespace emp;
using namespace std;
int bitlen = 32;
int party,port;
char *ip = "127.0.0.1"; 
template<typename T> 
vector<T> make_vector(size_t size) { 
return std::vector<T>(size); 
} 

template <typename T, typename... Args> 
auto make_vector(size_t first, Args... sizes) 
{ 
auto inner = make_vector<T>(sizes...); 
return vector<decltype(inner)>(first, inner); 
} 


int main(int argc, char** argv) {
parse_party_and_port(argv, &party, &port);
if(argc>3){
  ip=argv[3];
}
cout<<"Ip Address: "<<ip<<endl;
cout<<"Port: "<<port<<endl;
cout<<"Party: "<<(party==1? "CLIENT" : "SERVER")<<endl;
NetIO * io = new NetIO(party==ALICE ? nullptr : ip, port);
setup_semi_honest(io, party);


Integer w;
if ((party == BOB)) {
cout << ("Input w:") << endl;
}
/* Variable to read the clear value corresponding to the input variable w at (34,1-34,28) */
uint32_t __tmp_in_w;
if ((party == BOB)) {
cin >> __tmp_in_w;
}
w = Integer(bitlen, __tmp_in_w, BOB);

Integer x;
if ((party == BOB)) {
cout << ("Input x:") << endl;
}
/* Variable to read the clear value corresponding to the input variable x at (35,1-35,28) */
uint32_t __tmp_in_x;
if ((party == BOB)) {
cin >> __tmp_in_x;
}
x = Integer(bitlen, __tmp_in_x, BOB);

auto z = make_vector<Integer>( (int32_t)10);
if ((party == BOB)) {
cout << ("Input z:") << endl;
}
/* Variable to read the clear value corresponding to the input variable z at (36,1-36,32) */
uint32_t __tmp_in_z;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)10; i0++){
if ((party == BOB)) {
cin >> __tmp_in_z;
}
z[i0] = Integer(bitlen, __tmp_in_z, BOB);
}

auto a = make_vector<Integer>( (int32_t)10,  (int32_t)100);
if ((party == ALICE)) {
cout << ("Input a:") << endl;
}
/* Variable to read the clear value corresponding to the input variable a at (37,1-37,37) */
uint32_t __tmp_in_a;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)10; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)100; i1++){
if ((party == ALICE)) {
cin >> __tmp_in_a;
}
a[i0][i1] = Integer(bitlen, __tmp_in_a, ALICE);
}
}

Bit b;
if ((party == BOB)) {
cout << ("Input b:") << endl;
}
/* Variable to read the clear value corresponding to the input variable b at (38,1-38,26) */
bool __tmp_in_b;
if ((party == BOB)) {
cin >> __tmp_in_b;
}
b = Bit(__tmp_in_b, BOB);

Bit c;
if ((party == BOB)) {
cout << ("Input c:") << endl;
}
/* Variable to read the clear value corresponding to the input variable c at (39,1-39,26) */
bool __tmp_in_c;
if ((party == BOB)) {
cin >> __tmp_in_c;
}
c = Bit(__tmp_in_c, BOB);


finalize_semi_honest();
delete io; 
 
return 0;
}
