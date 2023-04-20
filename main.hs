
import Torch
import Control.Monad

printTensor :: Tensor -> IO ()
printTensor t = do
  putStr $ (show t) ++ "\n\n"

class Layer layer where
    numParams :: layer -> Int
    apply :: layer -> Tensor -> Tensor
    toList :: layer -> [Parameter]
    fromList :: layer -> [Tensor] -> IO layer

data LinearLayer = LinearLayer {
    w :: Parameter,
    b :: Parameter
}

mkLinear :: Int -> Int -> IO LinearLayer
mkLinear nIn nOut = do
        randomW <- randIO' [nIn, nOut]
        w1 <- makeIndependent $ randomW
        randomB <- randIO' [1, nOut]
        b1 <- makeIndependent $ randomB
        return (LinearLayer w1 b1)

instance Layer LinearLayer where

    numParams LinearLayer{w, b} = 2

    toList LinearLayer{w, b} = [w, b]

    fromList layer params = do
            w <- makeIndependent $ (params !! 0)
            b <- makeIndependent $ (params !! 1)
            return (LinearLayer w b)
    
    apply LinearLayer{w, b} input = 
            let v1 = (input `matmul` (toDependent w)) 
                v2 = (toDependent b)
            in v1 + v2

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

update :: MLP -> (IndependentTensor, IndependentTensor) -> IO MLP
update model (xi, target) = do

  let lr = 0.001 * (ones' [1])
      x = toDependent xi
      t = toDependent target
      o = apply model x
      loss = mean $ (o - t) ^ 2
      gradients = grad loss (toList model)
      output = mysgd lr (toList model) gradients

  model <- fromList model output
  printTensor loss
  return model

main :: IO ()
main = do
    
  xi <- makeIndependent $ ones' [10, 4]
  target <- makeIndependent $ ones' [10, 1]

  let dataList  = (Prelude.take 100 (Prelude.repeat (xi, target)))
  -- initialize model
  l <- sequence [ (mkLinear 4 3), (mkLinear 3 1) ] 
  -- train model
  output <- foldM update l dataList
  -- Evaluate final loss
  update output (dataList !! 0)
  putStrLn "done"

