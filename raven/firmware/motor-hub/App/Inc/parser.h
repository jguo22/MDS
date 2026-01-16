#ifndef INC_PARSER_H_
#define INC_PARSER_H_

#include "stdint.h"

 enum Parser_RW {
    Parser_Write = 0u,
    Parser_Read = 1u,
};

// 1 rw
// 4 type
// 3 chan

// rw bit 1
// type bits
// upper data


struct Parser_Handle {
    // configuration
    uint8_t (**reads)(uint8_t*, uint8_t); // read function pointer array
    void (**writes)(uint8_t*, uint8_t); // write function pointer array

    uint8_t len; // length of number of function pointers (num types)
    uint8_t typeBits; // bit length of (type field)

    // internal
    uint8_t typeMask; // mask for the type bits
};

void Parser_Init(struct Parser_Handle* handle);

uint8_t Parser_Handler(struct Parser_Handle* handle, uint8_t* data, uint8_t len);

#endif