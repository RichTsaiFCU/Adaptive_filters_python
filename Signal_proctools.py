import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

class SignalProcessor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def shift_buffer(self,buf, x):
          """
          Shifts the elements of a buffer and inserts a new value.
        
          Args:
              buf: The buffer to shift.
              x: The new value to insert.
        
          Returns:
              None
          """
        
            # buf[k] =buf[k-1]
            # ...
            # buf[2] = buf[1]
            # buf[1] = buf[0]
            # buf[0] = x
          for i in range(len(buf) - 1, 0, -1):
            buf[i] = buf[i - 1]
          buf[0] = x
    def fir_convolution(self,x, h ,buf):
          """
          Performs convolution of a signal x with a FIR filter h.
        
          Args:
              x: Input signal.
              h: FIR filter coefficients.
              buf: Buffer of x
            
          Returns:
              y: Output signal.
          """
          self.shift_buffer(buf, x)
          y = np.dot(buf,h)
          return y ,buf

    def lms_FIR_filter(self,x, d, N, mu):  #for FIR
          """
          Performs Least Mean Squares (LMS) filtering on a signal.
        
          Args:
              x: Input signal.
              d: Desired signal.
              N: Filter order.
              mu: Learning rate.
        
          Returns:
              y: Output signal.
              w: Final filter coefficients.
          """
          # Initialize filter coefficients and output signal
          w, buf = np.zeros(N), np.zeros(N)
        
          y,err = [],[]
        # Iterate through each sample
          for n in range(len(x)):
        
            # Calculate the output
            out, buf = self.fir_convolution(x[n], w ,buf)
        
            # Calculate the error
            e = d[n] - out
              
            # Update the filter coefficients
            norm_factor = np.dot(buf , buf) + 1e-8
            w += (mu / norm_factor) * e * buf
              
            mu *= 0.9995
            err.append(e)
            y.append(out)
          print("mu:",mu)
          return y, err, w
        
    def iir_convolution(self,x, a , b, bufa, bufb):
      """
      Performs convolution of a signal x with a IIR filter a & b.
    
      Args:
          x: Input signal.
          a, b: IIR filter coefficients.
          bufa & bufb: Buffer of each filters
    
      Returns:
          y: Output signal.
      """
    
      self.shift_buffer(bufb, x)
      y = np.dot(b, bufb) - np.dot(a[1:], bufa)
    
      # Update buffers
      self.shift_buffer(bufa, y)
    
      return y ,bufb, bufa
        
    def update_proportionate_weights(self,g, w, gamma, alpha=0.5):
        """
        Update proportionate weights for PNLMS filter.
    
        Args:
            g: Current proportionate weights.
            w: Current filter coefficients.
            gamma: Small constant to prevent division by zero.
            alpha: Smoothing factor to control the adaptation of the weights.
    
        Returns:
            Updated proportionate weights.
        """
        # Compute the maximum absolute coefficient value
        max_w = np.max(np.abs(w))
        # Avoid division by zero and stabilize the update
        max_w = max(max_w, 1e-8)
        # Update the weights based on the relative magnitude of each coefficient
        g = alpha * (np.abs(w) / max_w) + (1 - alpha) * g
        # Ensure all weights are above a small threshold
        g = np.maximum(g, gamma)
        return g
    
    def lms_IIR_filter(self,x, d, N, mu):
      """
      Performs Least Mean Squares (LMS) filtering on a signal.
    
      Args:
          x: Input signal.
          d: Desired signal.
          N: Filter order.
          mu: Learning rate.
    
      Returns:
          y: Output signal.
          w: Final filter coefficients.
      """
    
      # Initialize filter coefficients and output signal
      wb, wa, bufb, bufa = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N-1)
      # wb2, wa2, bufb2, bufa2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
      wa[0]=1
    
      # Iterate through each sample
      # for _ in range(1):
      y,err = [],[]
      # Initialize a weights vector for the proportionate update
    
      alpha=0.005
      gamma=0.1
      g = np.ones(N) *gamma
      g2 = np.ones(N) *gamma
      for n in range(len(x)):
    
        out, bufb, bufa = self.iir_convolution(x[n], wa , wb, bufa, bufb)
        # Compute error
        e = d[n] - out
    
        # Update coefficients
        norm_factor = np.dot(bufb, bufb)# + 1e-8
        wb = wb + mu  * e * bufb # / norm_factor
        norm_factor = np.dot(bufa, bufa) + 1e-8
        wa[1:] = wa[1:]- mu  * e * bufa #/ norm_factor
    
        mu *= 0.9995
    
        err.append(e)
        y.append(out)
      print("mu:",mu)
      return y, err, wa, wb
        
    def Second_order_Volterra_filter(self,x, d, M=128, L=10, mu1=0.2, mu2=0.2):
        #Initialized Filter
      nIters = min(len(x),len(d)) 
      L2=int(L*(L+1)/2)
      u = np.zeros(M)
      u2 = np.zeros((M,L2))
      w = np.zeros(M)
      h2 = np.zeros(L2)
      e = np.zeros(nIters)
      y=np.zeros(nIters)
      for n in range(nIters):
        #Processing filter's weighting factors  
        u[1:] = u[:-1]
        u[0] = x[n]
        u2_n = np.outer(u[:L],u[:L])
        u2_n = u2_n[np.triu_indices_from(u2_n)]
        u2[1:] = u2[:-1]
        u2[0] = u2_n

        #Computing output value
        x2 = np.dot(u2,h2)
        g = u + x2
        out = np.dot(w, g.T)

        #Defined the error
        e_n = d[n] - out

        #Update the Weighting of first filter as w
        w = w + mu1*e_n*g/(np.dot(g,g)+1e-8)

        #Update the Weighting of secondary filter h2
        grad_2 = np.dot(u2.T,w)
        h2 = h2 + mu2*e_n*grad_2/(np.dot(grad_2,grad_2)+1e-8)
          
        mu1 *= 0.9995
        mu2 *= 0.9995
        e[n] = e_n
        y[n]=out
      print("mu1:",mu1)
      print("mu2:",mu2)
      return y,e,w, h2

    def Third_order_Volterra_filter(self,x, d, M=128, L=10, mu1=0.2, mu2=0.2, mu3=0.2):
        nIters = min(len(x), len(d))
        L2 = int(L * (L + 1) / 2)
        L3 = int(L * (L + 1) * (L + 2) / 6)  # Calculate number of third-order combinations
        u = np.zeros(M)
        u2 = np.zeros((M, L2))
        u3 = np.zeros((M, L3))
        w = np.zeros(M)
        h2 = np.zeros(L2)
        h3 = np.zeros(L3)
        e = np.zeros(nIters)
        y = np.zeros(nIters)
    
        for n in range(nIters):
            # Update linear buffer
            u[1:] = u[:-1]
            u[0] = x[n]
    
            # Update quadratic buffer
            u2_n = np.outer(u[:L], u[:L])
            u2_n = u2_n[np.triu_indices_from(u2_n)]
            u2[1:] = u2[:-1]
            u2[0] = u2_n
    
            # Update cubic buffer
            u3_n = np.array([u[i] * u[j] * u[k] for i in range(L) for j in range(i, L) for k in range(j, L)])
            u3[1:] = u3[:-1]
            u3[0] = u3_n
    
            # Compute output from quadratic and cubic terms
            x2 = np.dot(u2, h2)
            x3 = np.dot(u3, h3)
    
            # Combined input
            g = u + x2 + x3
            out = np.dot(w, g.T)
            e_n = d[n] - out
    
            # Update weights
            w += mu1 * e_n * g / (np.dot(g, g) + 1e-8)
            grad_2 = np.dot(u2.T, w)
            h2 += mu2 * e_n * grad_2 / (np.dot(grad_2, grad_2) + 1e-8)
            grad_3 = np.dot(u3.T, w)
            h3 += mu3 * e_n * grad_3 / (np.dot(grad_3, grad_3) + 1e-8)
    
            mu1 *= 0.9995
            mu2 *= 0.9995
            mu3 *= 0.9995
            # Store error and output
            e[n] = e_n
            y[n] = out
    
        print("mu1:",mu1)
        print("mu2:",mu2)
        print("mu3:",mu3)
        return y, e ,w ,h2,h3

    def Fourth_order_Volterra_filter(self,x, d, M=128, L=10, mu1=0.72, mu2=0.002, mu3=0.002, mu4=0.002):
        nIters = min(len(x), len(d))
        L2 = int(L * (L + 1) / 2)
        L3 = int(L * (L + 1) * (L + 2) / 6)
        L4 = int(L * (L + 1) * (L + 2) * (L + 3) / 24)  # Calculate number of fourth-order combinations
        u = np.zeros(M)
        u2 = np.zeros((M, L2))
        u3 = np.zeros((M, L3))
        u4 = np.zeros((M, L4))
        w = np.zeros(M)
        h2 = np.zeros(L2)
        h3 = np.zeros(L3)
        h4 = np.zeros(L4)
        e = np.zeros(nIters)
        y = np.zeros(nIters)
    
        for n in range(nIters):
            # Update linear buffer
            u[1:] = u[:-1]
            u[0] = x[n]
    
            # Update quadratic buffer
            u2_n = np.outer(u[:L], u[:L])
            u2_n = u2_n[np.triu_indices_from(u2_n)]
            u2[1:] = u2[:-1]
            u2[0] = u2_n
    
            # Update cubic buffer
            u3_n = np.array([u[i] * u[j] * u[k] for i in range(L) for j in range(i, L) for k in range(j, L)])
            u3[1:] = u3[:-1]
            u3[0] = u3_n
    
            # Update quartic buffer
            u4_n = np.array([u[i] * u[j] * u[k] * u[l] for i in range(L) for j in range(i, L) for k in range(j, L) for l in range(k, L)])
            u4[1:] = u4[:-1]
            u4[0] = u4_n
    
            # Compute output from quadratic, cubic, and quartic terms
            x2 = np.dot(u2, h2)
            x3 = np.dot(u3, h3)
            x4 = np.dot(u4, h4)
    
            # Combined input
            g = u + x2 + x3 + x4
            out = np.dot(w, g.T)
            e_n = d[n] - out
    
            # Update weights
            w += mu1 * e_n * g / (np.dot(g, g) + 1e-8)
            grad_2 = np.dot(u2.T, w)
            h2 += mu2 * e_n * grad_2 / (np.dot(grad_2, grad_2) + 1e-8)
            grad_3 = np.dot(u3.T, w)
            h3 += mu3 * e_n * grad_3 / (np.dot(grad_3, grad_3) + 1e-8)
            grad_4 = np.dot(u4.T, w)
            h4 += mu4 * e_n * grad_4 / (np.dot(grad_4, grad_4) + 1e-8)
    
            mu1 *= 0.9995
            mu2 *= 0.9995
            mu3 *= 0.9995
            mu4 *= 0.9995
            # Store error and output
            e[n] = e_n
            y[n] = out
    
        print("mu1:",mu1)
        print("mu2:",mu2)
        print("mu3:",mu3)
        print("mu4:",mu4)
        return y, e, w, h2, h3, h4
    
    def custom_mse(self,y_true, y_pred):
        # Ensure both inputs are NumPy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate the squared error
        squared_error = np.square(y_true - y_pred)
    
        # Calculate the mean squared error
        mse = np.mean(squared_error)
    
        return mse
        
    def generate_noise_signal(self, length, gain):
        t = np.arange(length) / self.sample_rate
        x = np.random.randn(length)*gain
        return t, x
         
    def generate_bark_scale_signal(self, length, lowcut, highcut, Nband):
        """Generate a sinusoidal mixture signal using frequencies spaced across the Bark scale."""
        # Helper functions for frequency conversion
        def hz_to_bark(f):
            return 26.81 * (f / (1960 + f)) - 0.53

        def bark_to_hz(b):
            return 1960 * (b + 0.53) / (26.81 - (b + 0.53))

        bark_values = np.linspace(hz_to_bark(lowcut), hz_to_bark(highcut), Nband)
        frequencies = bark_to_hz(bark_values)
        t = np.arange(length) / self.sample_rate
        x = np.sum([np.sin(2 * np.pi * freq * t) for freq in frequencies], axis=0)
        x /= np.max(np.abs(x))
        return t, x

    def plot_signal(self, t, x, title='Signal Plot'):
        """Plot the given signal."""
        plt.figure(figsize=(10, 4))
        plt.plot(t, x)
        plt.title(title)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def generate_chirp_signal(self, duration, start_freq, end_freq, amplitude):
        """Generate a chirp signal from start_freq to end_freq over the specified duration."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        x = amplitude * chirp(t, f0=start_freq, f1=end_freq, t1=duration, method='linear')
        return t, x

# Example usage:
# processor = SignalProcessor(44100)
# t, x = processor.generate_bark_scale_signal(10000, 50, 20000, 10)
# processor.plot_signal(t, x, 'Sinusoidal Mixture Signal Across the Bark Scale')

# t, sweepX = processor.generate_chirp_signal(0.5, 20, 20000, 0.5)
# processor.plot_signal(t, sweepX, 'Linear Chirp Signal')
