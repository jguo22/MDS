#ifndef INC_COM_H_
#define INC_COM_H_

#include "main.h"
#include "stdbool.h"

enum Com_RxState {
    Com_RxStart = 0u,
    Com_RxHeader = 1u,
    Com_RxData = 2u,
    Com_RxCRC = 3u,
};

union Com_Header 
{
    struct {
        uint8_t pid : 2; // unused
        uint8_t len : 5;
        bool resp : 1; // basically ack but needs to be 1 for reading
    };
    uint8_t byte;
};

struct Com_Handle {
	// configuration
    CRC_HandleTypeDef* hcrc;

    uint8_t sByte; // start byte
    uint8_t maxData; // 0-32 bytes

    // direct data write to peripheral
    void (*send)(uint8_t*, uint8_t);

    // request bytes from peripheral
    void (*request)(uint8_t*, uint8_t);

    // if returned not 0 there is response in same buffer of that len
    uint8_t (*parse)(uint8_t*, uint8_t);

    // internal
    uint8_t* rxBuf;
    enum Com_RxState rxState;
    union Com_Header header;

    uint8_t* txBuf;
};

// init handle with config
void Com_Init(struct Com_Handle* handle);

// transmit a packet
void Com_Transmit(struct Com_Handle* handle, uint8_t* data, uint8_t len, bool resp);

// called when data received from receive request
void Com_Handler(struct Com_Handle* handle);

#endif