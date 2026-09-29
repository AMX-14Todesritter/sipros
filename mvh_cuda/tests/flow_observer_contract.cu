#include "../cuda/scoring.cuh"
#include <iostream>
__global__ void observe(mvh_flow::Counters *output) {
    const double peaks[]{100,200};
    const int classes[]{0,1};
    mvh_cuda::Scan scan{};scan.peaks=2;scan.lower=0;scan.upper=300;
    mvh_cuda::Config cfg{};cfg.flow=output;cfg.fragmentTolerance=0.5;
    mvh_cuda::IonCounter ions(scan,cfg,peaks,classes,nullptr);
    ions.observe(100); // class zero is geometric evidence
    ions.observe(200);
    ions.observe(100.5); // exact boundary excluded, as in original strict matcher
    ions.observe(150);
    ions.observe(-1); // emitted, but not queried
    output->v[0]=ions.offered;output->v[1]=ions.geometricHits;
}
int main() {
    mvh_cuda::Buffer<mvh_flow::Counters> out(1);
    observe<<<1,1>>>(out.p);mvh_cuda::synced();
    std::vector<mvh_flow::Counters> host;out.read(host);
    if(host[0].v[0]!=5 || host[0].v[1]!=2) return 1;
    std::cout<<"PASS: class-0 geometric evidence, strict boundary, misses and out-of-range ions\n";
}
