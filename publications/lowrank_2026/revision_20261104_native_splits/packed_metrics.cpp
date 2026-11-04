#include <cstdint>
#include <algorithm>
#include <omp.h>

extern "C" {
void bitcounts(const uint64_t* x, int64_t n, int words, uint16_t* counts, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i=0;i<n;++i) {
        unsigned s=0;
        for(int w=0;w<words;++w) s+=__builtin_popcountll(x[i*words+w]);
        counts[i]=static_cast<uint16_t>(s);
    }
}

void pair_tanimoto(const uint64_t* x,const uint16_t* counts,int words,
                   const int64_t* a,const int64_t* b,int64_t pairs,double* out,int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t p=0;p<pairs;++p) {
        unsigned intersection=0;
        for(int w=0;w<words;++w)
            intersection+=__builtin_popcountll(x[a[p]*words+w]&x[b[p]*words+w]);
        unsigned denom=counts[a[p]]+counts[b[p]]-intersection;
        out[p]=denom?static_cast<double>(intersection)/denom:0.;
    }
}

void closest_training(const uint64_t* x,const uint16_t* counts,const uint8_t* labels,
                      int64_t n,int words,const int64_t* queries,int64_t nq,double* out,int threads) {
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for(int64_t q=0;q<nq;++q) {
        const int64_t u=queries[q];const unsigned cu=counts[u];double best=0.;
        for(int64_t v=0;v<n;++v) {
            if(labels[v]!=0) continue;
            const unsigned cv=counts[v];
            const unsigned high=std::max(cu,cv),low=std::min(cu,cv);
            // Exact upper bound; pruning cannot remove a better neighbour.
            if(high && static_cast<double>(low)/high<=best) continue;
            unsigned intersection=0;
            for(int w=0;w<words;++w)
                intersection+=__builtin_popcountll(x[u*words+w]&x[v*words+w]);
            const unsigned denom=cu+cv-intersection;
            const double sim=denom?static_cast<double>(intersection)/denom:0.;
            best=std::max(best,sim);
            if(best>=1.) break;
        }
        out[q]=best;
    }
}
}
