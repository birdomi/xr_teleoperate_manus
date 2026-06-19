#include "stark-sdk.h"
#include "param.h"

#include "dds/Publisher.h"
#include "dds/Subscription.h"
#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/idl/go2/AudioData_.hpp>
#include <unitree/common/thread/recurrent_thread.hpp>

#include <iostream>
#include <csignal>
#include <chrono>
#include <thread>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <cstring>

std::atomic<bool> running(true);
void signal_handler(int) { running = false; }

// Brainco Hand ID
constexpr uint8_t L_id = 0x7e;
constexpr uint8_t R_id = 0x7f;
constexpr uint32_t baudrate = 460800;

// ------------------ Utility ------------------
std::vector<std::string> getAvailableSerialPorts() {
    std::vector<std::string> ports;
    for (const auto& entry : std::filesystem::directory_iterator("/dev")) {
        std::string path = entry.path().string();
        if (path.rfind("/dev/ttyUSB", 0) == 0) ports.push_back(path);
    }
    spdlog::info("Available Serial Ports: {}", fmt::join(ports, ", "));
    return ports;
}


// Hand connection struct
struct HandConnection {
    DeviceHandler* handle{nullptr};
    DeviceInfo* info{nullptr};
    std::string port;
};

// Try connecting a hand on a given port
HandConnection try_connect_hand(const std::string& port, uint8_t slave_id) {
    HandConnection ret;
    ret.port = port;

    DeviceHandler* handle = modbus_open(port.c_str(), baudrate);
    if (!handle) {
        spdlog::warn("Failed to open {} port", port);
        return ret;
    }

    DeviceInfo* info = stark_get_device_info(handle, slave_id);
    if (!info) {
        spdlog::warn("Failed to get device info from {} port", port);
        modbus_close(handle);
        return ret;
    }

    stark_set_finger_unit_mode(handle, slave_id, FINGER_UNIT_MODE_NORMALIZED);

    ret.handle = handle;
    ret.info = info;
    return ret;
}

// Find a hand from available ports given allowed SKUs
HandConnection find_hand(std::vector<std::string>& ports, uint8_t slave_id, const std::vector<SkuType>& allowed_skus, const std::string& hand_name) {
    for (const auto& port : ports) {
        spdlog::info("Trying port for {} hand from {} port", hand_name, port);
        auto conn = try_connect_hand(port, slave_id);
        if (!conn.handle) continue;

        if (std::find(allowed_skus.begin(), allowed_skus.end(), conn.info->sku_type) != allowed_skus.end()) {
            spdlog::info("{} hand bound to {} port", hand_name, port);
            ports.erase(std::remove(ports.begin(), ports.end(), port), ports.end());
            return conn;
        } else {
            modbus_close(conn.handle);
            spdlog::warn("Port {} is not {} hand (sku {}). Closed.", port, hand_name, (int)conn.info->sku_type);
        }
    }
    return {};
}

// ------------------ Hand Update Loop ------------------
void update_finger(DeviceHandler* handle, uint8_t slave_id,
                   unitree::robot::SubscriptionBase<unitree_go::msg::dds_::MotorCmds_>* lowcmd,
                   unitree::robot::RealTimePublisher<unitree_go::msg::dds_::MotorStates_>* lowstate,
                   unitree::robot::RealTimePublisher<unitree_go::msg::dds_::AudioData_>* tactile_pub,
                   const std::string& ns) {
    uint16_t positions[6], speeds[6];

    for (int i = 0; i < 6; ++i) {
        positions[i] = static_cast<uint16_t>(std::clamp(lowcmd->msg_.cmds()[i].q(), 0.f, 1.f) * 1000.f);
        speeds[i]    = static_cast<uint16_t>(std::clamp(lowcmd->msg_.cmds()[i].dq(), 0.f, 1.f) * 1000.f);
    }

    // Write commands
    stark_set_finger_positions_and_speeds(handle, slave_id, positions, speeds, 6);

    // Read status
    auto status = stark_get_motor_status(handle, slave_id);
    if (!status) return;

    for (int i = 0; i < 6; ++i) {
        lowstate->msg_.states()[i].q()        = status->positions[i] / 1000.f;
        lowstate->msg_.states()[i].dq()       = status->speeds[i] / 1000.f;
        lowstate->msg_.states()[i].tau_est()  = status->currents[i] / 1000.f;

        // if (status->currents[i] > 800) {
        //     spdlog::warn("{} finger {} over current: {} mA", ns, i, status->currents[i]);
        // }
    }
    lowstate->unlockAndPublish();
    free_motor_status_data(status);

    // 3. [신규 토픽] 촉각 데이터 채우기 (Tactile Data Only)
    // AudioStates_ 메시지 구조를 '그릇'으로만 빌려 씁니다.

    auto touch_data = stark_get_touch_status(handle, slave_id);
    
    // uint16_t tactile_array[15];
    // size_t data_size = 5 * sizeof(uint16_t);
    // size_t tactile_pub->msg_.states()[i].data().resize(data_size);
    // uint16_t* data_ptr = reinterpret_cast<uint16_t*>(tactile_pub->msg_.data().data());

    if (touch_data){
        // (1) 방 만들기: 루프 밖에서 한 번만 수행!
        // 5손가락 * 3센서 * 2바이트(uint16) = 총 30바이트
        size_t total_elements = 5 * 9;
        size_t total_bytes = total_elements * sizeof(uint16_t);
        tactile_pub->msg_.data().resize(total_bytes);

        // (2) 포인터 변환: 바이트 방을 'uint16 배열'처럼 쓰겠다고 선언
        uint16_t* data_ptr = reinterpret_cast<uint16_t*>(tactile_pub->msg_.data().data());

        for (int i = 0; i < 5; ++i) {
            // data_ptr을 배열처럼 사용 (인덱스 계산: i*3)
            
            // [i번째 손가락의 데이터 가져오기]
            uint16_t f1 = touch_data->items[i].normal_force1;
            uint16_t f2 = touch_data->items[i].normal_force2;
            uint16_t f3 = touch_data->items[i].normal_force3;
            uint16_t t1 = touch_data->items[i].normal_force1;
            uint16_t t2 = touch_data->items[i].normal_force2;
            uint16_t t3 = touch_data->items[i].normal_force3;
            uint16_t td1 = touch_data->items[i].normal_force1;
            uint16_t td2 = touch_data->items[i].normal_force2;
            uint16_t td3 = touch_data->items[i].normal_force3;
            
            // [배열에 넣기]
            data_ptr[i * 9 + 0] = f1; // 첫 번째 센서
            data_ptr[i * 9 + 1] = f2; // 두 번째 센서
            data_ptr[i * 9 + 2] = f3; // 세 번째 센서
            data_ptr[i * 9 + 3] = t1; // t첫 번째 센서
            data_ptr[i * 9 + 4] = t2; // t두 번째 센서
            data_ptr[i * 9 + 5] = t3; // t세 번째 센서
            data_ptr[i * 9 + 6] = td1; // td첫 번째 센서
            data_ptr[i * 9 + 7] = td2; // td두 번째 센서
            data_ptr[i * 9 + 8] = td3; // td세 번째 센서
        }
    }
    else{
        std::memset(data_ptr, 0, total_bytes);
    }

    // for (int i = 0; i < 6; ++i) {
    //     if (touch_data) {            
            // tactile_array[0] 
            // q 자리에 수직 압력 저장 (float 변환)
            // tactile_pub->msg_.states()[i].q()       = (float)touch_data->items[i].normal_force1;
            
            // // dq 자리에 전단력 저장 (선택 사항)
            // tactile_pub->msg_.states()[i].dq()      = (float)touch_data->items[i].tangential_force1;
            
            // // tau 자리에 근접 센서 저장 (선택 사항)
            // tactile_pub->msg_.states()[i].tau_est() = (float)touch_data->items[i].self_proximity1;
        // } else {
            // 데이터 없음
            // tactile_pub->msg_.states()[i].q() = 0.0f;
            // tactile_pub->msg_.states()[i].dq() = 0.0f;
            // tactile_pub->msg_.states()[i].tau_est() = 0.0f;
        // }
    // }
    tactile_pub->unlockAndPublish(); // 촉각 데이터 발송!
    if (touch_data) free_touch_finger_data(touch_data);
}

// Worker thread for each hand
void hand_worker(DeviceHandler* handle, uint8_t slave_id, const std::string& ns) {
    spdlog::info("🚀 Starting worker for {} (slave {})", ns, (int)slave_id);

    // [NEW] Enable Touch Sensors (bits: 0x1F = 0001 1111 -> // Enable all 5 fingers + palm
    stark_enable_touch_sensor(handle, slave_id, 0x3F);
    stark_calibrate_touch_sensor(handle, slave_id, 0x3F);
    spdlog::info("Enabled Touch Sensors for {}", ns);

    // DDS setup
    auto lowcmd   = std::make_shared<unitree::robot::SubscriptionBase<unitree_go::msg::dds_::MotorCmds_>>("rt/brainco/" + ns + "/cmd");
    lowcmd->msg_.cmds().resize(6);
    for (auto& finger : lowcmd->msg_.cmds()) finger.dq() = 1.;

    auto lowstate = std::make_unique<unitree::robot::RealTimePublisher<unitree_go::msg::dds_::MotorStates_>>("rt/brainco/" + ns + "/state");
    lowstate->msg_.states().resize(6);

    // 3. [NEW] 촉각 상태 Publisher (신규 토픽 이름: .../tactile)
    auto tactile_pub = std::make_unique<unitree::robot::RealTimePublisher<unitree_go::msg::dds_::AudioStates_>>("rt/brainco/" + ns + "/tactile");

    while (running) {
        auto start_time = std::chrono::high_resolution_clock::now();
        update_finger(handle, slave_id, lowcmd.get(), lowstate.get(), tactile_pub.get(), ns);
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        int sleep_us = 10000 - static_cast<int>(elapsed_us); // 100Hz
        if (sleep_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    }

    spdlog::info("Worker for {} exiting (closing handle)", ns);
    if (handle) modbus_close(handle);
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);

    auto vm = param::helper(argc, argv);
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network_interface"].as<std::string>());

    //init_cfg(StarkHardwareType::STARK_HARDWARE_TYPE_REVO2_BASIC,
    // init_cfg(StarkHardwareType::STARK_HARDWARE_TYPE_REVO2_TOUCH,
    init_cfg(StarkProtocolType::STARK_PROTOCOL_TYPE_MODBUS,
             LogLevel::LOG_LEVEL_ERROR);

    std::vector<std::string> available_ports = getAvailableSerialPorts();
    if (available_ports.empty()) {
        spdlog::warn("No ttyUSB serial ports found.");
        return 0;
    }

    HandConnection left_conn  = find_hand(available_ports, L_id, {SkuType::SKU_TYPE_SMALL_LEFT, SkuType::SKU_TYPE_MEDIUM_LEFT}, "left");
    HandConnection right_conn = find_hand(available_ports, R_id, {SkuType::SKU_TYPE_SMALL_RIGHT, SkuType::SKU_TYPE_MEDIUM_RIGHT}, "right");

    std::thread left_thread, right_thread;
    if (left_conn.handle)  left_thread  = std::thread(hand_worker, left_conn.handle, L_id, "left");
    if (right_conn.handle) right_thread = std::thread(hand_worker, right_conn.handle, R_id, "right");

    if (left_thread.joinable())  left_thread.join();
    if (right_thread.joinable()) right_thread.join();

    spdlog::info("exit.");
    return 0;
}
