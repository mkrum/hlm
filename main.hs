
import Torch
import Control.Monad

printTensor :: Tensor -> IO ()
printTensor t = putStrLn $ (show t)

class Layer layer where
    numParams :: layer -> Int
    apply :: layer -> Tensor -> Tensor
    toList :: layer -> [Parameter]
    fromList :: layer -> [Tensor] -> IO layer

data LinearLayer = LinearLayer {
    w :: Parameter,
    b :: Parameter,
    act :: Tensor -> Tensor
}

mkLinear :: Int -> Int -> (Tensor -> Tensor) -> IO LinearLayer
mkLinear nIn nOut actFn = do
        randomW <- randIO' [nIn, nOut]
        w1 <- makeIndependent $ randomW
        randomB <- randIO' [1, nOut]
        b1 <- makeIndependent $ randomB
        return (LinearLayer w1 b1 actFn)

instance Layer LinearLayer where

    numParams LinearLayer{w, b, act} = 2

    toList LinearLayer{w, b, act} = [w, b]

    fromList LinearLayer{w, b, act} params = do
            newW <- makeIndependent $ (params !! 0)
            newB <- makeIndependent $ (params !! 1)
            return (LinearLayer newW newB act)
    
    apply LinearLayer{w, b, act} input = 
            let v1 = (input `matmul` (toDependent w)) 
                v2 = (toDependent b)
            in act (v1 + v2)

type MLP = [LinearLayer]

instance Layer MLP where

    numParams = sum . map numParams 

    toList = concatMap toList 

    apply layers input = foldl (\i l -> apply l i) input layers
    
    fromList (layer:[]) tensors = sequence [fromList layer tensors]
    fromList layers tensors = do
            let layer = head layers 
                relParams = Prelude.take (numParams layer) tensors
            val <- fromList layer relParams
            tailValues <- fromList (tail layers) (drop (numParams layer) tensors)
            return (val:tailValues)
        
mysgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
mysgd lr params grads = 
        let updateParams p g = (toDependent p) - lr * g
        in zipWith updateParams params grads

update :: (Layer layer) => IO (Tensor, Tensor) -> ModelStep layer
update sampler model = do
        (x, y) <- sampler
        let lr = 0.01 * (ones' [1])
            o = apply model x
            loss = mean $ (o - y) ^ 2
            gradients = grad loss (toList model)
            output = mysgd lr (toList model) gradients
        printTensor loss
        model <- fromList model output
        return model

infiniteSampler :: [Int] -> (Tensor -> Tensor) -> IO (Tensor, Tensor)
infiniteSampler shape f = do
    x <- randIO' shape
    return (x, f x)

type ModelStep l = l -> IO l

applySteps :: (Layer layer) => [ModelStep layer] -> ModelStep layer 
applySteps updates model = foldM (\acc fn -> fn acc) model updates

lossEstimate :: (Layer layer) => IO (Tensor, Tensor) -> ModelStep layer
lossEstimate sampler model = do
        (x, y) <- sampler
        let o = apply model x
            loss = mean $ (o - y) ^ 2

        putStr "Evaluating: "
        printTensor loss
        return model

main :: IO ()
main = do

  -- initialize model
  model <- sequence $ [(mkLinear 1 16 Torch.tanh)] ++ (replicate 3 (mkLinear 16 16 Torch.tanh)) ++ [(mkLinear 16 1 id)]

  -- train model
  let sampler = infiniteSampler [100, 1] Torch.sin

  let evalStep = [lossEstimate sampler]
  let trainStep = [applySteps (replicate 10 (update sampler))]
  let updates = [evalStep, trainStep, evalStep]

  finalModel <- applySteps updates model
  return ()
