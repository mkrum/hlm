
import Torch

printTensor :: Tensor -> IO ()
printTensor t = do
  putStr $ (show t) ++ "\n\n"

class Layer layer where
        apply :: layer -> Tensor -> Tensor
        toList :: layer -> [Parameter]

data LinearLayer = LinearLayer {
    w :: Parameter,
    b :: Parameter
}

mkLinear :: Int -> Int -> IO LinearLayer
mkLinear nIn nOut = do
        w1 <- makeIndependent $ ones' [nIn, nOut]
        b1 <- makeIndependent $ ones' [1, nOut]
        let l = (LinearLayer w1 b1)
        return l

instance Layer LinearLayer where

    toList LinearLayer{w, b} = [w, b]

    apply LinearLayer{w, b} input = 
            let v1 = (input `matmul` (toDependent w)) 
                v2 = (toDependent b)
            in v1 + v2

type MLP = [LinearLayer]

instance Layer MLP where

    toList layers = foldl (\i l -> i ++ (toList l)) [] layers

    apply layers input = foldl (\i l -> apply l i) input layers


main :: IO ()
main = do

  xi <- makeIndependent $ ones' [10, 4]
  w1 <- makeIndependent $ ones' [4, 1]
  b1 <- makeIndependent $ ones' [1, 1]

  target <- makeIndependent $ ones' [10, 1]

  l <- sequence [ (mkLinear 4 3), (mkLinear 3 1) ] 

  let x = toDependent xi
      targets' = toDependent target
      h2 = apply l x
      loss = mean $ (h2 - targets') ^ 2
  let gradients = grad loss (toList l)

  printTensor (gradients !! 0)
  printTensor (gradients !! 1)
