#ifndef CPP_EXECUTOR_H
#define CPP_EXECUTOR_H

#include <string>
#include <vector>

class CppBinanceClient {
public:
    CppBinanceClient(const std::string& api_key, const std::string& api_secret);
    ~CppBinanceClient();

    std::string send_order(const std::string& symbol, const std::string& side, 
                           const std::string& type, double quantity, double price);

private:
    std::string api_key_;
    std::string api_secret_;
    
    // Simula generar firma HMAC-SHA256
    std::string generate_signature(const std::string& query_string);
};

#endif
