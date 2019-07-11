#include <cstdio>
#include <cstring>
using namespace std;
void mergesort(int l,int h,int a[]);
void combine(int l,int mid,int h,int a[]);

long long int ans=0;
int buffer[500001];

int main()
{
    int n;
    int arr[500001],i; ans=0;
    while(scanf("%d",&n)){
        if(n==0) return 0;
        ans = 0;
        memset(arr,0,sizeof(arr));
        for(int i=0;i<n;i++){
            scanf("%d",&arr[i]);}
        mergesort(0,n-1,arr);
        printf("%lld\n",ans);
    }
}
void mergesort(int l,int h,int a[])
{
    if(l==h) return;
    int mid=(l+h)/2;
    mergesort(l,mid,a);
    mergesort(mid+1,h,a);
    combine(l,mid,h,a);
}
void combine(int l,int mid,int h,int a[])
{
    int lcnt=l,hcnt=mid+1,bufcnt=0;
    while(lcnt<=mid && hcnt<=h){
        if(a[hcnt]<a[lcnt]){
            buffer[bufcnt++]=a[hcnt++];
            ans+=(mid-lcnt+1);
        }
        else buffer[bufcnt++]=a[lcnt++];
    }
    while(lcnt<=mid) buffer[bufcnt++]=a[lcnt++];
    while(hcnt<=h) buffer[bufcnt++]=a[hcnt++];
    for(bufcnt=0;l<=h;l++)
        a[l]=buffer[bufcnt++];
}
