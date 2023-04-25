
import Torch
import Control.Monad

printTensor :: Tensor -> IO ()
printTensor t = do
  putStrLn $ (show t)

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

    numParams layer = sum $ map numParams layer

    toList layers = foldl (\i l -> i ++ (toList l)) [] layers

    apply layers input = foldl (\i l -> apply l i) input layers

    fromList (layer:[]) tensors = sequence [fromList layer tensors]
    fromList layers tensors = do
            let layer = head layers 
                relParams = Prelude.take (numParams layer) tensors
            val <- fromList layer relParams
            tailValues <- fromList (tail layers) (drop (numParams layer) tensors)
            return (val:tailValues)
        
mysgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
mysgd lr params grads = map (\(p, g) -> (toDependent p) - lr * g) (zip params grads)

update :: MLP -> (Tensor, Tensor) -> IO MLP
update model (xi, target) = do

  let lr = 0.01 * (ones' [1])
      x = xi
      t = target
      o = apply model x
      loss = mean $ (o - t) ^ 2
      gradients = grad loss (toList model)
      output = mysgd lr (toList model) gradients

  printTensor loss
  model <- fromList model output
  return model


createBatch :: [Int] -> (Tensor -> Tensor) -> IO (Tensor, Tensor)
createBatch shape f = do
    x <- randIO' shape
    let y = f x
    return (x, y)


main :: IO ()
main = do

  dataList <- sequence $ [createBatch [100, 1] Torch.sin | _ <- [1..10]]

  -- initialize model
  l <- sequence $ [(mkLinear 1 16 Torch.tanh)] ++ [ (mkLinear 16 16 Torch.tanh) | _ <- [1..3]] ++ [(mkLinear 16 1 id)] 

  -- train model
  output <- foldM update l dataList
  return ()
