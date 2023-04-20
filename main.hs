
import Torch

printTensor :: Tensor -> IO ()
printTensor t = do
  putStr $ (show t) ++ "\n\n"

class Layer layer where
    numParams :: layer -> Int
    apply :: layer -> Tensor -> Tensor
    toList :: layer -> [Parameter]
    fromList :: layer -> [Tensor] -> IO layer

data LinearLayer = LinearLayer {
    n :: Int,
    w :: Parameter,
    b :: Parameter
}

mkLinear :: Int -> Int -> IO LinearLayer
mkLinear nIn nOut = do
        w1 <- makeIndependent $ ones' [nIn, nOut]
        b1 <- makeIndependent $ ones' [1, nOut]
        return (LinearLayer 2 w1 b1)

instance Layer LinearLayer where

    numParams LinearLayer{n, w, b} = n

    toList LinearLayer{n, w, b} = [w, b]

    fromList layer params = do
            w <- makeIndependent $ (params !! 0)
            b <- makeIndependent $ (params !! 1)
            return (LinearLayer 2 w b)
    
    apply LinearLayer{n, w, b} input = 
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

main :: IO ()
main = do

  xi <- makeIndependent $ ones' [10, 4]
  target <- makeIndependent $ ones' [10, 1]

  l <- sequence [ (mkLinear 4 3), (mkLinear 3 1) ] 

  let x = toDependent xi
      t = toDependent target
      o = apply l x
      loss = mean $ (o - t) ^ 2
      gradients = grad loss (toList l)
      output = mysgd ((zeros' [1]) + 0.001) (toList l) gradients
   
  printTensor loss

  l <- fromList l output

  let x = toDependent xi
      t = toDependent target
      o = apply l x
      loss = mean $ (o - t) ^ 2
      gradients = grad loss (toList l)
      output = mysgd (ones' [1]) (toList l) gradients

  printTensor loss
