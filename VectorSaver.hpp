#include <iostream>
#include <fstream>
#include <vector>

template <typename T>
class VectorSaver {
public:
    VectorSaver() {}

    template <typename U>
    bool SaveToFile(const std::string& filename, const std::vector<U>& data) const {
        std::ofstream outputFile(filename, std::ios::binary);
        if (outputFile.is_open()) {
            outputFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(U));
            outputFile.close();
            return true;
        } else {
            std::cerr << "Unable to open file for writing." << std::endl;
            return false;
        }
    }
};