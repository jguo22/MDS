#include "com.h"
#include "string.h"
#include "stdlib.h"

void Com_Init(struct Com_Handle* handle) {
    handle->rxBuf = malloc(handle->maxData+1);

    handle->rxState = Com_RxStart;

    handle->txBuf = malloc(handle->maxData+3);
    handle->txBuf[0] = handle->sByte;

    handle->request(handle->rxBuf, 1);
}

void Com_Transmit(struct Com_Handle* handle, uint8_t* data, uint8_t len, bool resp) {
    const union Com_Header header = {.resp = resp, .len = len};
    handle->txBuf[1] = header.byte;
    memcpy(&handle->txBuf[2], data, len);

    handle->txBuf[len+2] = HAL_CRC_Calculate(handle->hcrc, (uint32_t *)handle->txBuf, len+2);

    handle->send(handle->txBuf, len+3);
}

void Com_Handler(struct Com_Handle* handle) {
    switch (handle->rxState) {
    case Com_RxStart:
        if (handle->rxBuf[0] == handle->sByte) {
            HAL_CRC_Calculate(handle->hcrc, (uint32_t *)handle->rxBuf, 1);
            handle->rxState = Com_RxHeader;
        }

        handle->request(handle->rxBuf, 1);
        break;
    case Com_RxHeader:
        handle->header.byte = handle->rxBuf[0];
        HAL_CRC_Accumulate(handle->hcrc, (uint32_t *)handle->rxBuf, 1);

        if (handle->header.len) {
            handle->rxState = Com_RxData;
            // do this 1 byte forward to fit CRC
            // trust, makes sense for simplicity
            // avoids many memcpys
            handle->request(&handle->rxBuf[1], handle->header.len);
        }else{
            handle->rxState = Com_RxCRC;
            handle->request(handle->rxBuf, 1);
        }
        break;
    case Com_RxData:
        HAL_CRC_Accumulate(handle->hcrc, (uint32_t *)(&handle->rxBuf[1]), handle->header.len);
        handle->rxState = Com_RxCRC;

        handle->request(handle->rxBuf, 1);
        break;
    case Com_RxCRC:
        if (HAL_CRC_Accumulate(handle->hcrc, (uint32_t *)handle->rxBuf, 1)) {// wrong CRC
            Com_Transmit(handle, handle->rxBuf, 1, 0);
        } else {
            // now data is perfectly alligned for transmission
            const uint8_t respLen = handle->parse(&handle->rxBuf[1], handle->header.len);
            if (handle->header.resp) {
                // yay no memcpy here either
                Com_Transmit(handle, handle->rxBuf, respLen+1, 0);
            }
        }
        handle->rxState = Com_RxStart;
        handle->request(handle->rxBuf, 1);
        break;
    }
}
