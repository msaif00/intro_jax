import jax
import jax.numpy as jnp
import time

#PyTree for parameters
params = {
    'layer1': {
        'w': jax.random.normal(jax.random.PRNGKey(0), (784, 512)) /  jnp.sqrt(784), #Normalized
        'b': jnp.zeros(512)
    },
    'layer2': {
        'w': jax.random.normal(jax.random.PRNGKey(1), (512, 10)) /  jnp.sqrt(512),
        'b': jnp.zeros(10)
    }
}

def predict(params, inputs):
    z1 = jnp.dot(inputs,params['layer1']['w']) + params['layer1']['b']
    a1 = jax.nn.relu(z1)

    #output
    logits = jnp.dot(a1, params['layer2']['w']) + params['layer2']['b']
    return logits

@jax.jit
def loss_fn(params, x, y):
    preds = predict(params, x)
    return jnp.mean( (preds - y)**2 )


if __name__=="__main__":
    start = time.time()
    grad_fn = jax.grad(loss_fn)
    print(f"total time: {(time.time() - start):.6f}s")