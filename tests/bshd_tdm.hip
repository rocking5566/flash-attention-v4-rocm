#include "gfx1250.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define HIP(call) do { auto e=(call); if(e!=hipSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#call,hipGetErrorString(e)); std::exit(1); } } while(0)

template<int Pitch>
__global__ void probe(const __bf16* input,__bf16* output,int rows,int heads,int head) {
    __shared__ __align__(16) __bf16 tile[64*Pitch];
    if(threadIdx.x<32) {
        gfx1250::load_strided<64,0,(Pitch-128)*2>(tile,input+head*128,rows,heads*128);
        gfx1250::wait();
    }
    __syncthreads();
    for(int i=threadIdx.x;i<64*128;i+=128) output[i]=tile[i/128*Pitch+i%128];
}

int main() {
    int failures=0;
    for(int pitch: {136,144}) for(int heads: {3,8,513}) for(int rows: {1,17,47,64}) {
        const int head=heads-1;
        std::vector<__bf16> input(rows*heads*128),output(64*128);
        for(int r=0;r<rows;++r) for(int h=0;h<heads;++h) for(int d=0;d<128;++d)
            input[(r*heads+h)*128+d]=__bf16((r*137+h*71+d*13)%1021);
        __bf16 *din,*dout;
        HIP(hipMalloc(&din,input.size()*2)); HIP(hipMalloc(&dout,output.size()*2));
        HIP(hipMemcpy(din,input.data(),input.size()*2,hipMemcpyHostToDevice));
        if(pitch==136) probe<136><<<1,128>>>(din,dout,rows,heads,head);
        else probe<144><<<1,128>>>(din,dout,rows,heads,head);
        HIP(hipGetLastError());
        HIP(hipMemcpy(output.data(),dout,output.size()*2,hipMemcpyDeviceToHost));
        for(int r=0;r<64;++r) for(int d=0;d<128;++d) {
            float expected=r<rows ? float(input[(r*heads+head)*128+d]) : 0.f;
            if(float(output[r*128+d])!=expected) {
                if(failures++<8) std::printf("heads%d rows%d pitch%d r%d d%d got%g expected%g\n",
                    heads,rows,pitch,r,d,float(output[r*128+d]),expected);
            }
        }
        HIP(hipFree(din)); HIP(hipFree(dout));
    }
    std::printf("BSHD BF16 TDM load, strides, padding and zero-fill: %s (%d mismatches)\n",
                failures ? "FAIL" : "PASS",failures);
    return failures ? 1 : 0;
}
