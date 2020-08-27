#include "custom.hpp"
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <dirent.h>

using namespace std;

#define PI 3.14159265354

const int tab32[32] = {
    0,  9,  1, 10, 13, 21,  2, 29,
    11, 14, 16, 18, 22, 25,  3, 30,
    8, 12, 20, 28, 15, 17, 24,  7,
    19, 27, 23,  6, 26,  5,  4, 31
};

int log2_32 (unsigned int value) {
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[(unsigned int)(value*0x07C4ACDD) >> 27];
}

void log2_series(unsigned int value, vector<int>& param) {
    param.clear();
    while (value && log2_32(value)) {
        param.push_back(log2_32(value));
        value -= 2 << (param.back() - 1 );
    }
    if (value == 1) param.push_back(0);
}

__inline__ void get_dir(const float* const src_point, \
        const float rot_rad,
        float* src_res) {
    float sn = sin(rot_rad);
    float cs = cos(rot_rad);
    src_res[0] = src_point[0] * cs - src_point[1] * sn;
    src_res[1] = src_point[1] * sn + src_point[1] * cs;
}

__inline__ void get_3rd_point(const cv::Point2f& a,\
        const cv::Point2f& b, cv::Point2f& out) {
    out.x = b.x + b.y - a.y;
    out.y = b.y + a.x - b.x;
}

void get_affine_transform(
        float* mat_data, 
        const float* const center, \
        const float* const scale, \
        const float* const shift, 
        const float rot, \
        const int output_h, const int output_w, 
        const bool inv) {
    float rot_rad = rot * PI / 180.;
    float src_p[2] = {0, scale[0] * -0.5};
    float dst_p[2] = {0, output_w * -0.5};
    float src_dir[2], dst_dir[2];
    get_dir(src_p, rot_rad, src_dir);
    get_dir(dst_p, rot_rad, dst_dir);

    cv::Point2f src[3], dst[3];
    src[0] = cv::Point2f(center[0] + scale[0] * shift[0],
            center[1] + scale[1] * shift[1]);
    src[1] = cv::Point2f(center[0] + src_dir[0] + scale[0] * shift[0], center[1] + src_dir[1] + scale[1] * shift[1]);
    dst[0] = cv::Point2f(output_w * 0.5, output_h * 0.5);
    dst[1] = cv::Point2f(output_w * 0.5 + dst_dir[0], output_h*0.5 + dst_dir[1]);
    get_3rd_point(dst[0], dst[1], dst[2]);
    get_3rd_point(src[0], src[1], src[2]);
    cv::Mat warp_mat;
    if (inv) {
        warp_mat = getAffineTransform(dst, src);
    } else {
        warp_mat = getAffineTransform(src, dst);
    }
    //return warp_mat;
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            mat_data[i * 3 + j] = warp_mat.at<double>(i, j);
        }
    }
}


vector<string> split_str(string& str, string pattern) {
    vector<string> res;
    if (pattern.empty()) return res;
    size_t start = 0, index = str.find(pattern, 0);
    while (index != str.npos) {
        if (start != index) res.push_back(str.substr(start, index - start));
        start = index + 1;
        index = str.find(pattern, start);

    }
    if (!str.substr(start).empty()) res.push_back(str.substr(start));
    return res;
}

void makedirs(const char* dir) {
    char str[512];
    strncpy(str, dir, 512);
    int len = strlen(str);
    for(int i = 0 ; i < len; ++i) {
        if(str[i] == '/') {
            str[i] = '\0';
            if (access(str, 0) != 0) mkdir(str, 0777);
            str[i] = '/';
        }
    }
    if(len > 0 && access(str, 0) != 0) mkdir(str, 0777);
}

