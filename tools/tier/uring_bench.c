// O_DIRECT read ceiling: raw io_uring (no liburing) vs pread threads. CPU only.
//   cc -O2 -pthread -o uring_bench uring_bench.c
//   uring_bench FILE MODE RSIZE_KIB QD SECONDS [SPAN_GIB] [SEQ]
//   MODE: uring | threads | sqpoll ; QD = in-flight reads (uring) or threads ; SEQ=1 sequential chunks of RSIZE
// Prints: mode rsize qd GB/s IOPS p50_ms p99_ms
#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/io_uring.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int cmpd(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y;}
static uint64_t rs=88172645463325252ULL;
static inline uint64_t rnd(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return rs;}

static int fd; static size_t rsize, span_blk; static int qd, seq; static double secs;
static size_t next_off(uint64_t *ctr){
  if(seq){uint64_t i=__atomic_fetch_add(ctr,1,__ATOMIC_RELAXED);return (i*rsize)%(span_blk*4096-rsize);}
  return (rnd()%span_blk)*4096;
}
static uint64_t gctr;
// ---- threads
static double *lat_t[256]; static size_t nlat_t[256]; static uint64_t bytes_t[256];
static void *tw(void*arg){
  long ti=(long)arg; void*buf=aligned_alloc(4096,rsize); uint64_t r=88172645463325252ULL^(ti*2654435761u);
  lat_t[ti]=malloc(sizeof(double)*4000000); double end=now()+secs;
  while(now()<end){
    size_t off; if(seq){uint64_t i=__atomic_fetch_add(&gctr,1,__ATOMIC_RELAXED);off=(i*rsize)%(span_blk*4096-rsize);}
    else{r^=r<<13;r^=r>>7;r^=r<<17;off=(r%span_blk)*4096;}
    double t0=now(); ssize_t n=pread(fd,buf,rsize,off); double t1=now();
    if(n<=0){perror("pread");break;}
    if(nlat_t[ti]<4000000)lat_t[ti][nlat_t[ti]++]=t1-t0; bytes_t[ti]+=n;
  }
  return 0;
}
// ---- io_uring
static int sys_setup(unsigned e,struct io_uring_params*p){return syscall(__NR_io_uring_setup,e,p);}
static int sys_enter(int f,unsigned s,unsigned m,unsigned fl){return syscall(__NR_io_uring_enter,f,s,m,fl,NULL,0);}
int main(int argc,char**argv){
  if(argc<6){fprintf(stderr,"usage\n");return 2;}
  const char*path=argv[1],*mode=argv[2]; rsize=atol(argv[3])*1024; qd=atoi(argv[4]); secs=atof(argv[5]);
  double span_gib=argc>6?atof(argv[6]):8; seq=argc>7?atoi(argv[7]):0;
  fd=open(path,O_RDONLY|O_DIRECT); if(fd<0){perror("open");return 1;}
  off_t fs=lseek(fd,0,SEEK_END); size_t span=(size_t)(span_gib*(1<<30)); if(span>fs)span=fs;
  span_blk=(span-rsize)/4096;
  double t0=now(),tot_s; uint64_t bytes=0; size_t nl=0; double*lat;
  if(!strcmp(mode,"threads")){
    pthread_t th[256]; for(long i=0;i<qd;i++)pthread_create(&th[i],0,tw,(void*)i);
    for(int i=0;i<qd;i++)pthread_join(th[i],0);
    tot_s=now()-t0; lat=malloc(sizeof(double)*4000000*1);
    for(int i=0;i<qd;i++){bytes+=bytes_t[i];for(size_t j=0;j<nlat_t[i]&&nl<4000000;j++)lat[nl++]=lat_t[i][j];}
  } else {
    struct io_uring_params p; memset(&p,0,sizeof p);
    if(!strcmp(mode,"sqpoll")){p.flags=IORING_SETUP_SQPOLL;p.sq_thread_idle=2000;}
    int rfd=sys_setup(qd*2,&p); if(rfd<0){perror("io_uring_setup");return 1;}
    size_t sqsz=p.sq_off.array+p.sq_entries*sizeof(unsigned), cqsz=p.cq_off.cqes+p.cq_entries*sizeof(struct io_uring_cqe);
    void*sq=mmap(0,sqsz>cqsz?sqsz:cqsz,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,rfd,IORING_OFF_SQ_RING);
    void*cq=sq; // single mmap
    struct io_uring_sqe*sqes=mmap(0,p.sq_entries*sizeof(struct io_uring_sqe),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,rfd,IORING_OFF_SQES);
    unsigned*sqtail=sq+p.sq_off.tail,*sqmask=sq+p.sq_off.ring_mask,*sqarr=sq+p.sq_off.array,*sqflags=sq+p.sq_off.flags;
    unsigned*cqhead=cq+p.cq_off.head,*cqtail=cq+p.cq_off.tail,*cqmask=cq+p.cq_off.ring_mask; struct io_uring_cqe*cqes=cq+p.cq_off.cqes;
    char**bufs=malloc(qd*sizeof(char*)); double*tstart=malloc(qd*sizeof(double)); for(int i=0;i<qd;i++)bufs[i]=aligned_alloc(4096,rsize);
    lat=malloc(sizeof(double)*4000000); unsigned tail=*sqtail;
    #define SUBMIT(i) do{ size_t off=next_off(&gctr); unsigned idx=tail&*sqmask; struct io_uring_sqe*s=&sqes[idx]; memset(s,0,sizeof *s); \
      s->opcode=IORING_OP_READ; s->fd=fd; s->addr=(uint64_t)bufs[i]; s->len=rsize; s->off=off; s->user_data=i; sqarr[idx]=idx; tail++; tstart[i]=now(); }while(0)
    unsigned pend=0; for(int i=0;i<qd;i++){SUBMIT(i);pend++;}
    __atomic_store_n(sqtail,tail,__ATOMIC_RELEASE);
    int sqp=!strcmp(mode,"sqpoll");
    if(sqp){ if(__atomic_load_n(sqflags,__ATOMIC_ACQUIRE)&IORING_SQ_NEED_WAKEUP) sys_enter(rfd,0,0,IORING_ENTER_SQ_WAKEUP);} else sys_enter(rfd,pend,0,0);
    double end=now()+secs; int inflight=qd; t0=now();
    while(inflight>0){
      unsigned h=*cqhead; unsigned t=__atomic_load_n(cqtail,__ATOMIC_ACQUIRE);
      if(h==t){ if(!sqp) sys_enter(rfd,0,1,IORING_ENTER_GETEVENTS); continue; }
      unsigned sub=0;
      while(h!=t){
        struct io_uring_cqe*c=&cqes[h&*cqmask]; int i=c->user_data;
        if(c->res<=0){fprintf(stderr,"cqe res %d\n",c->res);return 1;}
        double tn=now(); if(nl<4000000)lat[nl++]=tn-tstart[i]; bytes+=c->res; h++;
        if(tn<end){SUBMIT(i);sub++;} else inflight--;
      }
      __atomic_store_n(cqhead,h,__ATOMIC_RELEASE);
      if(sub){__atomic_store_n(sqtail,tail,__ATOMIC_RELEASE);
        if(sqp){ if(__atomic_load_n(sqflags,__ATOMIC_ACQUIRE)&IORING_SQ_NEED_WAKEUP) sys_enter(rfd,0,0,IORING_ENTER_SQ_WAKEUP);} else sys_enter(rfd,sub,0,0);}
    }
    tot_s=now()-t0;
  }
  qsort(lat,nl,sizeof(double),cmpd);
  printf("%s rsize=%zuKiB qd=%d %s: %.3f GB/s  %.0f IOPS  p50 %.2f ms  p99 %.2f ms\n",mode,rsize/1024,qd,seq?"seq":"rand",
    bytes/tot_s/1e9,nl/tot_s,lat[nl/2]*1e3,lat[(size_t)(nl*0.99)]*1e3);
  return 0;
}
