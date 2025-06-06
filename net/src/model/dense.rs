use tensor::tensor::Tensor2;
use tensor::tensor_ops::matmul::MatMul;
use tensor::tensor_ops::elemwise::{ElemAdd, ElemSub};
use tensor::tensor_ops::const_ops::ConstMul;
use tensor::storage::naive_cpu::NaiveCpu;
use tensor::storage::HasStorage;
use rand::Rng;

pub struct Dense<const IN: usize, const OUT: usize>
where
    [f32; IN * OUT]:,
    [f32; OUT]:,
    [f32; 1 * OUT]:,
    [(); IN * OUT]:,
    [(); OUT]:,
    [(); 1 * OUT]:,
{
    pub w: Tensor2<f32, IN, OUT, NaiveCpu>,
    pub b: Tensor2<f32, 1, OUT, NaiveCpu>,
    pub grad_w: Tensor2<f32, IN, OUT, NaiveCpu>,
    pub grad_b: Tensor2<f32, 1, OUT, NaiveCpu>,
}

impl<const IN: usize, const OUT: usize> Dense<IN, OUT>
where
    [(); IN * OUT]:,
    [(); OUT]:,
    [(); 1 * OUT]:,
{
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / IN as f32).sqrt();

        let mut w_data = [0.0; IN * OUT];
        for val in w_data.iter_mut() {
            *val = rng.gen_range(-scale..scale);
        }

        Self {
            w: Tensor2::new(w_data),
            b: Tensor2::new([0.0; 1 * OUT]),
            grad_w: Tensor2::new([0.0; IN * OUT]),
            grad_b: Tensor2::new([0.0; 1 * OUT]),
        }
    }

    pub fn forward<const BATCH: usize>(
        &self,
        x: &Tensor2<f32, BATCH, IN, NaiveCpu>,
    ) -> Tensor2<f32, BATCH, OUT, NaiveCpu>
    where
        [(); BATCH * IN]:,
        [(); IN * OUT]:,
        [(); BATCH * OUT]:,
    {
        // Wx 
        let xw = x.clone() * self.w.clone();
        let xw_data = xw.as_slice();
        let b_data = self.b.as_slice();

        let mut out_data = <NaiveCpu as HasStorage<f32, { BATCH * OUT }>>::storage_uninit();
        let out_slice = <NaiveCpu as HasStorage<f32, { BATCH * OUT }>>::as_mut_slice(&mut out_data);

        // Wx + b -- add b here
        for i in 0..BATCH {
            for j in 0..OUT {
                out_slice[i * OUT + j] = xw_data[i * OUT + j] + b_data[j];
            }
        }

        Tensor2 {
            storage: out_data,
            _p: core::marker::PhantomData,
        }
    }

    pub fn backward<const BATCH: usize>(
        &mut self,
        x: &Tensor2<f32, BATCH, IN, NaiveCpu>,
        grad_out: &Tensor2<f32, BATCH, OUT, NaiveCpu>,
    ) -> Tensor2<f32, BATCH, IN, NaiveCpu>
    where
        [(); BATCH * IN]:,
        [(); IN * OUT]:,
        [(); BATCH * OUT]:,
    {
        // dl/dW = x^T * grad_out
        let x_data = x.as_slice();
        let grad_out_data = grad_out.as_slice();
        let mut grad_w_data = <NaiveCpu as HasStorage<f32, { IN * OUT }>>::storage_uninit();
        let grad_w_slice = <NaiveCpu as HasStorage<f32, { IN * OUT }>>::as_mut_slice(&mut grad_w_data);
        
        for i in 0..IN {
            for j in 0..OUT {
                let mut acc = 0.0;
                for b in 0..BATCH {
                    acc += x_data[b * IN + i] * grad_out_data[b * OUT + j];
                }
                grad_w_slice[i * OUT + j] = acc;
            }
        }
        self.grad_w = Tensor2 {
            storage: grad_w_data,
            _p: core::marker::PhantomData,
        };

        // dL/db = sum(grad_out, axis=0)
        let mut grad_b_data = <NaiveCpu as HasStorage<f32, { 1 * OUT }>>::storage_uninit();
        let grad_b_slice = <NaiveCpu as HasStorage<f32, { 1 * OUT }>>::as_mut_slice(&mut grad_b_data);
        
        for j in 0..OUT {
            let mut acc = 0.0;
            for b in 0..BATCH {
                acc += grad_out_data[b * OUT + j];
            }
            grad_b_slice[j] = acc;
        }
        self.grad_b = Tensor2 {
            storage: grad_b_data,
            _p: core::marker::PhantomData,
        };

        // dL/dx = grad_out * W^T
        let w_data = self.w.as_slice();
        let mut grad_x_data = <NaiveCpu as HasStorage<f32, { BATCH * IN }>>::storage_uninit();
        let grad_x_slice = <NaiveCpu as HasStorage<f32, { BATCH * IN }>>::as_mut_slice(&mut grad_x_data);

        for b in 0..BATCH {
            for i in 0..IN {
                let mut acc = 0.0;
                for j in 0..OUT {
                    acc += grad_out_data[b * OUT + j] * w_data[i * OUT + j];
                }
                grad_x_slice[b * IN + i] = acc;
            }
        }

        Tensor2 {
            storage: grad_x_data,
            _p: core::marker::PhantomData,
        }
    }
    
}
