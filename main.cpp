#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
using namespace std;
class tensorTransform;
class ReLU;
class Sigmoid;
class tensor{
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
    tensor(const std::vector<size_t>& shape_,const std::vector<double>& values_) {
        shape = shape_;
        owner = true;
        size_t total = tamaño(shape_);
        if (values_.size() != total)
            throw invalid_argument("Shape y values no coinciden");
        if (total == 0) return;
        data = new double[total];
        for (size_t i = 0; i < total; i++) {
            data[i] = values_[i];
        }
    }
    void setOwner (bool owner_) {owner = owner_;}
    static size_t tamaño ( const std :: vector < size_t >& shape_) {
        size_t total = 1;
        for (auto x : shape_)
            total *= x;
        return total;
    }
    static tensor zeros (const std :: vector < size_t >& shape_) {
        const std::vector values_(tamaño(shape_), 0.0);
        return {shape_, values_};
    }
    static tensor ones (const std :: vector < size_t >& shape_) {
        const std::vector values_(tamaño(shape_), 1.0);
        return {shape_, values_};
    }
    static tensor random (const std :: vector < size_t >& shape_, double min, double max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dist(min, max);
        std::vector<double> values_(tamaño(shape_));
        for (double& value : values_)
            value = dist(gen);
        return {shape_, values_};
    }
    static tensor arange(int min, int max) {
        std::vector<double> values_(max-min);
        for (int i=0; i < max-min; i++)
            values_[i] = i+min;
        std::vector<size_t> shape_(1, max-min);
        return {shape_, values_};
    }
    static tensor concat(const std::vector<tensor>& tensores, size_t x) {
        if (tensores.empty())
            throw invalid_argument("No hay tensores para concatenar");
        if (x >= tensores[0].shape.size())
            throw invalid_argument("Dimension invalida");
        for (const auto& tensor1:tensores) {
            const tensor& seguro = tensor1;
            for (const auto & tensor2: tensores) {
                if (seguro.shape.size() != tensor2.shape.size())
                    throw invalid_argument("Las dimensiones no son compatibles");
                for (size_t i=0; i < seguro.shape.size(); i++)
                    if (i != x and seguro.shape[i] != tensor2.shape[i])
                        throw invalid_argument("Las dimensiones no son compatibles");
            }
        }
        size_t total_nuevo=0;
        for (const auto& tensor_:tensores)
            total_nuevo += tamaño(tensor_.shape);
        std::vector<double> values_(total_nuevo);
        std::vector<size_t> shape_ = tensores[0].shape;
        for (const auto& tensor_:tensores)
            shape_[x] += tensor_.shape[x];
        shape_[x] -= tensores[0].shape[x];
        if (x == 0) {
            size_t seguidor = 0;
            for (const auto& tensor_:tensores) {
                size_t total = tamaño(tensor_.shape);
                for (size_t i = 0; i < total; i++)
                    values_[i+seguidor] = tensor_.data[i];
                seguidor += total;
            }
        } else {
            size_t seguidor = 0;
            for (size_t i = 0; i < tensores[0].shape[0]; i++) {
                for (const auto& tensor_:tensores) {
                    size_t total = tensor_.shape[1];
                    for (size_t j = 0; j < total; j++)
                        values_[seguidor+j] = tensor_.data[j+i*total];
                    seguidor += total;
                }
            }
        }
        return {shape_, values_};
    }
    tensor (const tensor& other) ; // Copia
    tensor ( tensor&& other ) noexcept ; // Movimiento
    tensor& operator =( const tensor& other) {
        size_t total = tamaño(other.shape);
        if (this != &other) {
            auto* new_data = new double[total];
            std::copy(other.data,other.data+total, new_data);
            if (owner) delete[] data;
            data = new_data;
            shape = other.shape;
            owner = true;
        }
        return *this;
    } // Asignacion copia
    tensor& operator =( tensor&& other ) noexcept {
        if (this != &other) {
            if (owner) delete[] data;
            data = other.data;
            shape = other.shape;
            owner = other.owner;
            other.shape={};
            other.data = nullptr;
            other.owner = false;
        }
        return *this;
    } // Asignacion movimiento
    [[nodiscard]] tensor apply (const tensorTransform& transform) const;
    [[nodiscard]] tensor view(const std::vector<size_t>& new_shape) const {
        if (tamaño(new_shape) != tamaño(shape))
            throw invalid_argument("Las dimensiones no son compatibles");
        if (new_shape.size() > 3)
            throw invalid_argument("Maximo 3 dimensiones");
        tensor nuevo;
        nuevo.data = this->data;
        nuevo.shape = new_shape;
        nuevo.owner = false;
        return nuevo;
    }
    [[nodiscard]] tensor unsqueeze(int x) const{
        if (x>this->shape.size())
            throw invalid_argument("La dimension no es valida");
        if (this->shape.size()+1 > 3)
            throw invalid_argument("Maximo 3 dimensiones");
        tensor nuevo;
        nuevo.data = this->data;
        nuevo.shape = this->shape;
        nuevo.shape.insert(nuevo.shape.begin()+x, 1);
        nuevo.owner = false;
        return nuevo;
    }
    friend tensor dot(const tensor& a, const tensor& b);
    friend tensor matmul(const tensor& a, const tensor& b);
    ~tensor() {if (owner) delete[] data;}
};
class tensorTransform {
public :
    [[nodiscard]] virtual tensor apply ( const tensor & t ) const = 0;
    virtual ~ tensorTransform () = default ;
};
class ReLU: public tensorTransform {
public :
    [[nodiscard]] tensor apply(const tensor& t) const override {
        size_t total = tensor::tamaño(t.shape);
        std::vector<double> values_(total);
        for (size_t i=0; i<total; i++) {
            if (t.data[i] <= 0) values_[i] = 0;
            else values_[i] = t.data[i];
        }
        return {t.shape, values_};
    }
};
class Sigmoid: public tensorTransform {
public :
    [[nodiscard]] tensor apply ( const tensor & t ) const override {
        size_t total = tensor::tamaño(t.shape);
        std::vector<double> values_(total);
        for (size_t i=0; i<total; i++) {
            double x = t.data[i];
            values_[i] = 1.0 / (1.0 + exp(-x));
        }
        return {t.shape, values_};
    }
};
tensor::tensor(const tensor& other): shape(other.shape), owner(true) {
    size_t total = tamaño(other.shape);
    if (total == 0) return;
    data = new double[total];
    for (size_t i = 0; i < total; i++) {
        data[i] = other.data[i];
    }
}
tensor::tensor(tensor &&other) noexcept {
    owner = other.owner;
    other.owner = false;
    data = other.data;
    other.data = nullptr;
    shape = other.shape;
    other.shape = {};
}
tensor tensor::apply (const tensorTransform& transform) const {
    return transform.apply(*this);
}
tensor operator+(const tensor& t_izq, const tensor& t_der) {
    if (t_izq.shape != t_der.shape) throw invalid_argument("Las dimensiones no son compatibles");
    size_t total = tensor::tamaño(t_izq.shape);
    std::vector<double> values_(total);
    for (size_t i=0; i<total; i++)
        values_[i] = t_izq.data[i] + t_der.data[i];
    return {t_izq.shape, values_};
}
tensor operator-(const tensor& t_izq, const tensor& t_der) {
    if (t_izq.shape != t_der.shape) throw invalid_argument("Las dimensiones no son compatibles");
    size_t total = tensor::tamaño(t_izq.shape);
    std::vector<double> values_(total);
    for (size_t i=0; i<total; i++)
        values_[i] = t_izq.data[i] - t_der.data[i];
    return {t_izq.shape, values_};
}
tensor operator*(const tensor& t_izq, const tensor& t_der) {
    if (t_izq.shape != t_der.shape) throw invalid_argument("Las dimensiones no son compatibles");
    size_t total = tensor::tamaño(t_izq.shape);
    std::vector<double> values_(total);
    for (size_t i=0; i<total; i++)
        values_[i] = t_izq.data[i] * t_der.data[i];
    return {t_izq.shape, values_};
}
tensor operator*(const tensor& t_izq, double d) {
    size_t total = tensor::tamaño(t_izq.shape);
    std::vector<double> values_(total);
    for (size_t i=0; i<total; i++)
        values_[i] = t_izq.data[i] * d;
    return {t_izq.shape, values_};
}
tensor dot (const tensor& a,const tensor& b) {
    if (a.shape != b.shape) throw invalid_argument("Las dimensiones no son compatibles");
    double suma = 0.0;
    size_t total = tensor::tamaño(a.shape);
    for (size_t i = 0; i < total; i++)
        suma += a.data[i] * b.data[i];
    return tensor({1}, {suma});
}
tensor matmul (const tensor& a,const tensor& b) {
    if (a.shape.size() != 2 or b.shape.size()!= 2) throw invalid_argument("Dimension diferente a 2D");
    if (a.shape[1] != b.shape[0]) throw invalid_argument("Las dimensiones no son compatibles");
    std::vector<double> values_(a.shape[0]*b.shape[1]);
    for (size_t i=0; i<a.shape[0]; i++) {
        for (size_t j=0; j<b.shape[1]; j++) {
            values_[j+i*b.shape[1]] = 0;
            for (size_t k=0; k<a.shape[1]; k++) {
                values_[j+i*b.shape[1]] += a.data[k+i*a.shape[1]]*b.data[k*b.shape[1]+j];
            }
        }
    }
    return {{a.shape[0],b.shape[1]},values_};
}
int main() {
    tensor X = tensor::random({1000, 20, 20}, 0.0, 1.0);

    tensor Xflat = X.view({1000, 400});

    tensor W1 = tensor::random({400, 100}, -1.0, 1.0);
    tensor Z1 = matmul(Xflat, W1);

    tensor b1 = tensor::random({1, 100}, -1.0, 1.0);
    std::vector<tensor> bias1_list(1000, b1);
    tensor B1 = tensor::concat(bias1_list, 0);
    tensor Z1b = Z1 + B1;

    ReLU relu;
    tensor A1 = Z1b.apply(relu);

    tensor W2 = tensor::random({100, 10}, -1.0, 1.0);
    tensor Z2 = matmul(A1, W2);

    tensor b2 = tensor::random({1, 10}, -1.0, 1.0);
    std::vector<tensor> bias2_list(1000, b2);
    tensor B2 = tensor::concat(bias2_list, 0);
    tensor Z2b = Z2 + B2;

    Sigmoid sigmoid;
    tensor Y = Z2b.apply(sigmoid);

    return 0;
}