#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

int demo_done=0;
const int net_width=416;        
const int net_height=416;
const int img_channels=3;
cv::Mat img[3];    //IMREAD_COLOR=1 default  BGR
int buff_index=0;
int outputs_index=0;
cv::VideoCapture cap;
rknn_context ctx;

rknn_input inputs[1];
rknn_output outputs[2][3];
rknn_perf_detail perf_detail;
rknn_perf_run perf_run;
rknn_tensor_attr outputs_attr[3];

Scalar colorArray[10]={
        Scalar(139,0,0,255),
        Scalar(139,0,139,255),
        Scalar(0,0,139,255),
        Scalar(0,100,0,255),
        Scalar(139,139,0,255),
        Scalar(209,206,0,255),
        Scalar(0,127,255,255),
        Scalar(139,61,72,255),
        Scalar(0,255,0,255),
        Scalar(255,0,0,255),
};

static string labels[80]={"person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light","fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant","bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet ","tvmonitor","laptop","mouse","remote ","keyboard ","cell phone","microwave ","oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush "};

static int GRID0=13;
static int GRID1=26;
static int GRID2=52;
static int nclasses=80;
static int nyolo=3; //n yolo layers;
static int nanchor=3; //n anchor per yolo layer

static int nboxes_0=GRID0*GRID0*nanchor;
static int nboxes_1=GRID1*GRID1*nanchor;
static int nboxes_2=GRID2*GRID2*nanchor;

static int nboxes_total=nboxes_0+nboxes_1+nboxes_2;

float OBJ_THRESH=0.2;
float DRAW_CLASS_THRESH=0.2;
float NMS_THRESH=0.4;  //darknet demo nms=0.4


typedef struct{
        float x,y,w,h;
}box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float objectness;
    int sort_class;
} detection;

detection* dets=0;

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets.prob);
    }
    free(dets);
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int netw, int neth, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / netw;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / neth;
    return b;
}

detection *get_network_boxes(float *predictions, int netw,int neth,int GRID,int* masks, float* anchors, int box_off)
{
        int lw=GRID;
        int lh=GRID;
        int nboxes=GRID*GRID*nanchor;
        int LISTSIZE=1+4+nclasses;

        for(int n=0;n<nanchor;n++){
                int index=n*lw*lh*LISTSIZE;
                int index_end=index+2*lw*lh;
                for(int i=index;i<index_end;i++)
                        predictions=1./(1.+exp(-predictions));                        
        }

        for(int n=0;n<nanchor;n++){
                int index=n*lw*lh*LISTSIZE+4*lw*lh;
                int index_end=index+(1+nclasses)*lw*lh;
                for(int i=index;i<index_end;i++){
                        predictions=1./(1.+exp(-predictions));               
                }
        }

        int count=box_off;
        for(int i=0;i<lw*lh;i++){
                int row=i/lw;
                int col=i%lw;
                for(int n=0;n<nanchor;n++){
                        int box_loc=n*lw*lh+i;  
                        int box_index=n*lw*lh*LISTSIZE+i;            
                        int obj_index=box_index+4*lw*lh;
                        float objectness=predictions[obj_index];
                        if(objectness<OBJ_THRESH) continue;
                        dets[count].objectness=objectness;
                        dets[count].classes=nclasses;
                        dets[count].bbox=get_yolo_box(predictions,anchors,masks[n],box_index,col,row,lw,lh,netw,neth,lw*lh);
                        for(int j=0;j<nclasses;j++){
                                int class_index=box_index+(5+j)*lw*lh;
                                float prob=objectness*predictions[class_index];
                                dets[count].prob[j]=prob;
                        }
                        ++count;
                }
        }
        //cout<<"count: "<<count-box_off<<"\n";
    return dets;
}

detection* outputs_transform(rknn_output rknn_outputs[], int net_width, int net_height){
        float* output_0=(float*)rknn_outputs[0].buf;
        float* output_1=(float*)rknn_outputs[1].buf;
        float* output_2=(float*)rknn_outputs[2].buf;
        int masks_0[3] = {6, 7, 8};
        int masks_1[3] = {3, 4, 5};
        int masks_2[3] = {0, 1, 2};
        //float anchors[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
        float anchors[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};

        get_network_boxes(output_0,net_width,net_height,GRID0,masks_0,anchors,0);
        get_network_boxes(output_1,net_width,net_height,GRID1,masks_1,anchors,nboxes_0);
        get_network_boxes(output_2,net_width,net_height,GRID2,masks_1,anchors,nboxes_0+nboxes_1);
        return dets;

}

float overlap(float x1,float w1,float x2,float w2){
        float l1=x1-w1/2;
        float l2=x2-w2/2;
        float left=l1>l2? l1:l2;
        float r1=x1+w1/2;
        float r2=x2+w2/2;
        float right=r1<r2? r1:r2;
        return right-left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}
int do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets.objectness == 0){
            detection swap = dets;
            dets = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;
        //cout<<"total after OBJ_THRESH: "<<total<<"\n";

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets.sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets.prob[k] == 0) continue;
            box a = dets.bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
        return total;
}

int do_nms_obj(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
        //cout<<"k: "<<k<<"\n";
    for(i = 0; i <= k; ++i){
        if(dets.objectness == 0){
            detection swap = dets;
            dets = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;
        //cout<<"total after OBJ_THRESH: "<<total<<"\n";
    for(i = 0; i < total; ++i){
        dets.sort_class = -1;
    }

    qsort(dets, total, sizeof(detection), nms_comparator);  
    for(i = 0; i < total; ++i){
        if(dets.objectness == 0) continue;
        box a = dets.bbox;
        for(j = i+1; j < total; ++j){
            if(dets[j].objectness == 0) continue;
            box b = dets[j].bbox;
            if (box_iou(a, b) > thresh){
                dets[j].objectness = 0;
                for(k = 0; k < classes; ++k){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
        return total;
}

int draw_image(cv::Mat img,detection* dets,int total,float thresh){
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        for(int i=0;i<total;i++){
                char labelstr[4096]={0};
                int class_=-1;
                int topclass=-1;
                float topclass_score=0;
                if(dets.objectness==0) continue;
                for(int j=0;j<nclasses;j++){
                        if(dets.prob[j]>thresh){
                                if(topclass_score<dets.prob[j]){
                                        topclass_score=dets.prob[j];
                                        topclass=j;
                                }
                                if(class_<0){
                                        strcat(labelstr,labels[j].data());
                                        class_=j;
                                }
                                else{
                                        strcat(labelstr,",");
                                        strcat(labelstr,labels[j].data());
                                }
                                printf("%s: %.02f%%\n",labels[j].data(),dets.prob[j]*100);
                        }
                }

                if(class_>=0){
                        box b=dets.bbox;
                        int x1 =(b.x-b.w/2.)*img.cols;
                        int x2=(b.x+b.w/2.)*img.cols;
                        int y1=(b.y-b.h/2.)*img.rows;
                        int y2=(b.y+b.h/2.)*img.rows;

            if(x1  < 0) x1  = 0;
            if(x2> img.cols-1) x2 = img.cols-1;
            if(y1 < 0) y1 = 0;
            if(x2 > img.rows-1) x2 = img.rows-1;
                        std::cout << labels[topclass] << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

            rectangle(img, Point(x1, y1), Point(x2, y2), colorArray[class_%10], 3);
            putText(img, labelstr, Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
            }
                }
        
        //imwrite("out.jpg",img);
        return 0;
}

void *display_in_thread(void *ptr){
        double start_time,end_time;
        start_time=what_time_is_it_now();
        cv::Mat img_show;
        img_show=img[(buff_index)%3];
        //cv::resize(img[(buff_index)%3],img_show,cv::Size(600,600),0,0,cv::INTER_LINEAR);
        cv::imshow("yolov3_tiny",img_show);
        int c = cv::waitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        DRAW_CLASS_THRESH += .02;
    } else if (c == 84) {
        DRAW_CLASS_THRESH -= .02;
        if(DRAW_CLASS_THRESH <= .02) DRAW_CLASS_THRESH = .02;
    } else if (c == 83) {
        NMS_THRESH += .02;
    } else if (c == 81) {
        NMS_THRESH -= .02;
        if(NMS_THRESH <= .0) NMS_THRESH = .0;
    }
        end_time=what_time_is_it_now();
        cout<<"display thread time:"<<(end_time-start_time)*1000<<" ms\n";
        return 0;

}

void *fetch_in_thread(void *ptr){
        double start_time,end_time;
        start_time=what_time_is_it_now();
        cv::Mat frame;
        cap>>frame;
    if(!frame.data) {
        printf("fetch frame fail!\n");
        return 0;
    }
    if(frame.cols != net_width || frame.rows != net_height)
        cv::resize(frame, frame, cv::Size(net_width, net_height), (0, 0), (0, 0), cv::INTER_LINEAR);
        //BGR->RGB
        cv::cvtColor(frame, img[buff_index], cv::COLOR_BGR2RGB);
        //cv::imshow("camera",frame);
        end_time=what_time_is_it_now();
        cout<<"fetch thread time:"<<(end_time-start_time)*1000<<" ms\n";
        return 0;
}

void *process_in_thread(void *ptr){
        double start_time,end_time;
        start_time=what_time_is_it_now();
        int nboxes_left=0;
        for(int i=0;i<nboxes_total;++i){
                dets.objectness=0;
        }
        outputs_transform(outputs[(outputs_index+1)%2],net_width,net_height);
        nboxes_left=do_nms_sort(dets,nboxes_total,nclasses,NMS_THRESH);
        //nboxes_left=do_nms_obj(dets,nboxes_total,nclasses,NMS_THRESH);
        draw_image(img[(buff_index+2)%3],dets,nboxes_left,DRAW_CLASS_THRESH);
        rknn_outputs_release(ctx,3,outputs[(outputs_index+1)%2]);
        end_time=what_time_is_it_now();
        cout<<"process thread time:"<<(end_time-start_time)*1000<<" ms\n";
        return 0;

}

void *detect_in_thread(void *ptr){
        double start_time,end_time;
        start_time=what_time_is_it_now();
        inputs[0].buf = img[(buff_index+2)%3].data;
        int ret = rknn_inputs_set(ctx, 1, inputs);
        if(ret < 0) {
                printf("rknn_input_set fail! ret=%d\n", ret);
                goto Error;
        }
        //run
    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        goto Error;
    }
        ret = rknn_outputs_get(ctx, 3, outputs[outputs_index], NULL);
        if(ret < 0) {
                printf("rknn_outputs_get fail! ret=%d\n", ret);
                goto Error;
        }
        //if(outputs[outputs_index][0].size != outputs_attr[0].n_elems*sizeof(float)|| outputs[outputs_index][1].size != outputs_attr[1].n_elems*sizeof(float)|| outputs[outputs_index][2].size != outputs_attr[2].n_elems*sizeof(float)){
        //        printf("outputs_size!=outpus_attr\n");
        //        goto Error;
        //}

        //ret = rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail,sizeof(rknn_perf_detail));
        //printf("%s\n", perf_detail.perf_data);

        //ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(rknn_perf_run));
        //printf("NPU run time: %ld us\n", perf_run.run_duration);
        end_time=what_time_is_it_now();
        cout<<"detect thread time:"<<(end_time-start_time)*1000<<" ms\n";
        return 0;
        Error:
                if(ctx) rknn_destroy(ctx);
        return 0;
}

int main(int argc, char** argb){
        const char *model_path = "./model_out/yolov3.rknn";
        cv::Mat frame;
        cv::namedWindow("yolov3");
        //cv::namedWindow("camera");
        double begin,end,real_fps;
        int frame_count=0;

        //alloc total dets
        dets =(detection*) calloc(nboxes_total,sizeof(detection));
        for(int i=0;i<nboxes_total;++i){
                dets.prob=(float*) calloc(nclasses,sizeof(float));
        }
        
    // Load model
    FILE *fp = fopen(model_path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);   
    int model_len = ftell(fp);   
    void *model = malloc(model_len);
    fseek(fp, 0, SEEK_SET);   
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", model_path);
        free(model);
        return -1;
    }
        cout<<"model load done\n";

        int ret=0;
        rknn_input_output_num io_num;

        //init
        ret=rknn_init(&ctx,model,model_len,RKNN_FLAG_PRIOR_MEDIUM);
        //ret=rknn_init(&ctx,model,model_len,RKNN_FLAG_PRIOR_MEDIUM | RKNN_FLAG_COLLECT_PERF_MASK);
        if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        goto Error;
    }
        printf("rknn_init done\n");

        //rknn input output num
        ret = rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io_num,sizeof(io_num));
        if(ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        goto Error;
    }
        cout<<"io_num query done\n";
        cout<<"input num: "<<io_num.n_input<<";  ";
        cout<<"output num: "<<io_num.n_output<<";  \n";
        //rknn inputs
        inputs[0].index = 0;
        inputs[0].size = net_width * net_height * img_channels;
        inputs[0].pass_through = false;         
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;

        //rknn outputs
        outputs[0][0].want_float = true;
        outputs[0][0].is_prealloc = false;
        outputs[0][1].want_float = true;
        outputs[0][1].is_prealloc = false;
        outputs[0][2].want_float = true;
        outputs[0][2].is_prealloc = false;
        
        outputs[1][0].want_float = true;
        outputs[1][0].is_prealloc = false;
        outputs[1][1].want_float = true;
        outputs[1][1].is_prealloc = false;
        outputs[1][2].want_float = true;
        outputs[1][2].is_prealloc = false;
        //rknn outputs_attr
        outputs_attr[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
    if(ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        goto Error;
    }
    outputs_attr[1].index = 1;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[1]), sizeof(outputs_attr[1]));
    if(ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        goto Error;
    }
        outputs_attr[2].index = 2;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[2]), sizeof(outputs_attr[2]));
    if(ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        goto Error;
    }

        pthread_t detect_thread;
        pthread_t process_thread;
        cap.open("/dev/video0");
        if(!cap.isOpened()){
                return -1;
        }
        cout<<"camera open done\n";
        
        cap>>frame;
        if(!frame.data) {
        printf("fetch frame fail!\n");
        return 0;
    }
    if(frame.cols != net_width || frame.rows != net_height)
        cv::resize(frame, frame, cv::Size(net_width, net_height), (0, 0), (0, 0), cv::INTER_LINEAR);
    //BGR->RGB
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    img[0]=frame.clone();
    img[1]=frame.clone();
    img[2]=frame.clone();
    detect_in_thread(0);
    cout<<"first detect done\n";

    begin=what_time_is_it_now();
    while(!demo_done){
            double start_time,end_time,fps;
            start_time=what_time_is_it_now();
            clock_t start,end;
            outputs_index=(outputs_index+1)%2;
            buff_index=(buff_index+1)%3;
        
            if(pthread_create(&process_thread,0,process_in_thread,0)) {
                    printf("Thread process creation failed");
                    break;
           }
           if(pthread_create(&detect_thread,0,detect_in_thread,0)) {
                printf("Thread detect creation failed");
                break;
           }