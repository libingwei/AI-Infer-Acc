#include <iostream>
#include <trt_utils/trt_common.h>

int main() {
    // Sanity: access a symbol from trt_utils headers
    TrtLogger logger; // should compile & link if package works
    std::cout << "trt_utils consumer ok\n";
    return 0;
}
