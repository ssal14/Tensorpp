#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
using namespace std;

class tensorTransform;
class ReLU;
class Sigmoid;

class tensor {
    double* data;
    std::vector<size_t> shape;
    bool owner;

    friend tensor operator+(const tensor&, const tensor&);
    friend tensor operator-(const tensor&, const tensor&);
    friend tensor operator*(const tensor&, const tensor&);
    friend tensor operator*(const tensor&, double);
    friend tensor dot(const tensor& a, const tensor& b);
    friend tensor matmul(const tensor& a, const tensor& b);

public:
    friend class ReLU;
    friend class Sigmoid;

    tensor(): data(nullptr), owner(true) {}

    tensor(const std::vector<size_t>& shape_, const std::vector<double>& values_) {
        shape = shape_;
        owner = true;

        size_t total = tamaño(shape_);
        if (values_.size() != total)
            throw invalid_argument("Shape y values no coinciden");

        if (total == 0) {
            data = nullptr;
            return;
        }

        data = new double[total];
        for (size_t i = 0; i < total; i++)
            data[i] = values_[i];
    }

    void setOwner(bool owner_) { owner = owner_; }

    static size_t tamaño(const std::vector<size_t>& shape_) {
        size_t total = 1;
        for (auto x : shape_)
            total *= x;
        return total;
    }

    static tensor zeros(const std::vector<size_t>& shape_) {
        const std::vector<double> values_(tamaño(shape_), 0.0);
        return {shape_, values_};
    }

    static tensor ones(const std::vector<size_t>& shape_) {
        const std::vector<double> values_(tamaño(shape_), 1.0);
        return {shape_, values_};
    }

    static tensor random(const std::vector<size_t>& shape_, double min, double max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(min, max);

        std::vector<double> values_(tamaño(shape_));
        for (double& value : values_)
            value = dist(gen);

        return {shape_, values_};
    }

    static tensor arange(int min, int max) {
        if (max <= min)
            throw invalid_argument("Rango invalido");

        std::vector<double> values_(max - min);
        for (int i = 0; i < max - min; i++)
            values_[i] = i + min;

        std::vector<size_t> shape_(1, max - min);
        return {shape_, values_};
    }

    static tensor concat(const std::vector<tensor>& tensores, size_t x) {
        if (tensores.empty())
            throw invalid_argument("No hay tensores");

        if (x >= tensores[0].shape.size())
            throw invalid_argument("Dimension invalida");

        // validar compatibilidad
        for (const auto& t1 : tensores) {
            for (const auto& t2 : tensores) {
                if (t1.shape.size() != t2.shape.size())
                    throw invalid_argument("Dimensiones incompatibles");

                for (size_t i = 0; i < t1.shape.size(); i++) {
                    if (i != x && t1.shape[i] != t2.shape[i])
                        throw invalid_argument("Dimensiones incompatibles");
                }
            }
        }

        std::vector<size_t> shape_ = tensores[0].shape;
        shape_[x] = 0;
        for (const auto& t : tensores)
            shape_[x] += t.shape[x];

        size_t total_nuevo = tamaño(shape_);
        std::vector<double> values_(total_nuevo);

        size_t offset = 0;
        for (const auto& t : tensores) {
            size_t total = tamaño(t.shape);
            for (size_t i = 0; i < total; i++)
                values_[offset + i] = t.data[i];
            offset += total;
        }

        return {shape_, values_};
    }

    tensor(const tensor& other) : shape(other.shape), owner(true) {
        size_t total = tamaño(other.shape);
        if (total == 0) {
            data = nullptr;
            return;
        }
        data = new double[total];
        for (size_t i = 0; i < total; i++)
            data[i] = other.data[i];
    }

    tensor(tensor&& other) noexcept {
        data = other.data;
        shape = other.shape;
        owner = other.owner;

        other.data = nullptr;
        other.shape = {};
        other.owner = false;
    }

    tensor& operator=(const tensor& other) {
        if (this != &other) {
            size_t total = tamaño(other.shape);
            double* new_data = new double[total];

            for (size_t i = 0; i < total; i++)
                new_data[i] = other.data[i];

            if (owner) delete[] data;

            data = new_data;
            shape = other.shape;
            owner = true;
        }
        return *this;
    }

    tensor& operator=(tensor&& other) noexcept {
        if (this != &other) {
            if (owner) delete[] data;

            data = other.data;
            shape = other.shape;
            owner = other.owner;

            other.data = nullptr;
            other.shape = {};
            other.owner = false;
        }
        return *this;
    }

    tensor apply(const tensorTransform& transform) const;

    tensor view(const std::vector<size_t>& new_shape) const {
        if (tamaño(new_shape) != tamaño(shape))
            throw invalid_argument("Dimensiones incompatibles");

        if (new_shape.size() > 3)
            throw invalid_argument("Max 3D");

        tensor nuevo;
        nuevo.data = this->data;
        nuevo.shape = new_shape;
        nuevo.owner = false;
        return nuevo;
    }

    tensor unsqueeze(int x) const {
        if (x < 0 || x > shape.size())
            throw invalid_argument("Dimension invalida");

        if (shape.size() + 1 > 3)
            throw invalid_argument("Max 3D");

        tensor nuevo;
        nuevo.data = this->data;
        nuevo.shape = shape;
        nuevo.shape.insert(nuevo.shape.begin() + x, 1);
        nuevo.owner = false;
        return nuevo;
    }

    ~tensor() {
        if (owner) delete[] data;
    }
};

class tensorTransform {
public:
    virtual tensor apply(const tensor& t) const = 0;
    virtual ~tensorTransform() = default;
};

class ReLU : public tensorTransform {
public:
    tensor apply(const tensor& t) const override {
        size_t total = tensor::tamaño(t.shape);
        std::vector<double> values_(total);

        for (size_t i = 0; i < total; i++)
            values_[i] = max(0.0, t.data[i]);

        return {t.shape, values_};
    }
};

class Sigmoid : public tensorTransform {
public:
    tensor apply(const tensor& t) const override {
        size_t total = tensor::tamaño(t.shape);
        std::vector<double> values_(total);

        for (size_t i = 0; i < total; i++)
            values_[i] = 1.0 / (1.0 + exp(-t.data[i]));

        return {t.shape, values_};
    }
};

tensor tensor::apply(const tensorTransform& transform) const {
    return transform.apply(*this);
}

tensor operator+(const tensor& a, const tensor& b) {
    if (a.shape != b.shape)
        throw invalid_argument("Dimensiones incompatibles");

    size_t total = tensor::tamaño(a.shape);
    std::vector<double> values_(total);

    for (size_t i = 0; i < total; i++)
        values_[i] = a.data[i] + b.data[i];

    return {a.shape, values_};
}

tensor operator-(const tensor& a, const tensor& b) {
    if (a.shape != b.shape)
        throw invalid_argument("Dimensiones incompatibles");

    size_t total = tensor::tamaño(a.shape);
    std::vector<double> values_(total);

    for (size_t i = 0; i < total; i++)
        values_[i] = a.data[i] - b.data[i];

    return {a.shape, values_};
}

tensor operator*(const tensor& a, const tensor& b) {
    if (a.shape != b.shape)
        throw invalid_argument("Dimensiones incompatibles");

    size_t total = tensor::tamaño(a.shape);
    std::vector<double> values_(total);

    for (size_t i = 0; i < total; i++)
        values_[i] = a.data[i] * b.data[i];

    return {a.shape, values_};
}

tensor operator*(const tensor& a, double d) {
    size_t total = tensor::tamaño(a.shape);
    std::vector<double> values_(total);

    for (size_t i = 0; i < total; i++)
        values_[i] = a.data[i] * d;

    return {a.shape, values_};
}

tensor dot(const tensor& a, const tensor& b) {
    if (a.shape != b.shape)
        throw invalid_argument("Dimensiones incompatibles");

    double suma = 0;
    size_t total = tensor::tamaño(a.shape);

    for (size_t i = 0; i < total; i++)
        suma += a.data[i] * b.data[i];

    return tensor({1}, {suma});
}

tensor matmul(const tensor& a, const tensor& b) {
    if (a.shape.size() != 2 or b.shape.size() != 2)
        throw invalid_argument("Solo 2D");

    if (a.shape[1] != b.shape[0])
        throw invalid_argument("Dimensiones incompatibles");

    std::vector<double> values_(a.shape[0] * b.shape[1]);

    for (size_t i = 0; i < a.shape[0]; i++) {
        for (size_t j = 0; j < b.shape[1]; j++) {
            double sum = 0;
            for (size_t k = 0; k < a.shape[1]; k++) {
                sum += a.data[i * a.shape[1] + k] *
                       b.data[k * b.shape[1] + j];
            }
            values_[i * b.shape[1] + j] = sum;
        }
    }

    return {{a.shape[0], b.shape[1]}, values_};
}