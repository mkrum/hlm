
import Torch

printTensor :: Tensor -> IO ()
printTensor t = do
  putStr $ (show t) ++ "\n\n"

class Layer layer where
        apply :: layer -> Tensor -> Tensor
        toList :: layer -> [Parameter]
        fromList :: [Parameter] -> layer

data LinearLayer = LinearLayer {
    n :: Int, 
    w :: Parameter,
    b :: Parameter
}

instance Layer LinearLayer where
        toList LinearLayer{n, w, b} = [w, b]
        fromList params = LinearLayer 2 (params !! 0) (params !! 1)
        apply LinearLayer{n, w, b} input = 
                let v1 = (input `matmul` (toDependent w)) 
                    v2 = (toDependent b)
                in v1 + v2

type MLP = [LinearLayer]

instance Layer MLP where
        toList layers = foldl (\i l -> i ++ (toList l)) [] layers
        fromList params = []
        apply layers input = foldl (\i l -> apply l i) input layers

main :: IO ()
main = do

  xi <- makeIndependent $ ones' [10, 4]
  w1 <- makeIndependent $ ones' [4, 1]
  b1 <- makeIndependent $ ones' [1, 1]

  let l = [fromList [w1, b1] :: LinearLayer] :: MLP

  let x = toDependent xi
      h2 = apply l x
      loss = mean $ (h2 - 0) ^ 2
  let gradients = grad loss (toList l)

  printTensor (gradients !! 0)
  printTensor (gradients !! 1)
