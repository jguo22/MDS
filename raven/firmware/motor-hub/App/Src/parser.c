#include "parser.h"



void Parser_Init(struct Parser_Handle* handle) {
    handle->typeMask = (1<<handle->typeBits)-1;
}

uint8_t Parser_Handler(struct Parser_Handle* handle, uint8_t* data, uint8_t len) {
    if (!len) return 0; // invalid or watchdog

    const uint8_t type = (data[0]>>(7-handle->typeBits))&handle->typeMask;
    if (type >= handle->len) return 0; // invalid
    
    if (data[0]>>7 == Parser_Read) {
        if (handle->reads[type]) return (*handle->reads[type])(data, len); // Minus 1 for header
        return 0;
    }
    if (handle->writes[type]) (*handle->writes[type])(data, len); // Minus 1 for header
    return 0;
}