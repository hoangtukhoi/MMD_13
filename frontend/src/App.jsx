import { useState, useEffect } from 'react'
import axios from 'axios'
import { ShoppingCart, Plus, Trash2, Sparkles, Box } from 'lucide-react'
import { useStore } from './store'
import './index.css'

const API_URL = 'http://localhost:8000/api'

function App() {
  const [seedProducts, setSeedProducts] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [loadingInitial, setLoadingInitial] = useState(true)
  const [loadingRecs, setLoadingRecs] = useState(false)
  
  const cart = useStore((state) => state.cart)
  const addToCart = useStore((state) => state.addToCart)
  const removeFromCart = useStore((state) => state.removeFromCart)

  // Fetch 20 Seed Products (Cold Start - Big Data)
  useEffect(() => {
    const fetchSeed = async () => {
      try {
        const res = await axios.get(`${API_URL}/seed`)
        if(res.data && res.data.data) {
           setSeedProducts(res.data.data)
        }
      } catch (err) {
        console.error("Error loading seed data:", err)
      } finally {
        setLoadingInitial(false)
      }
    }
    fetchSeed()
  }, [])

  // Fetch Recommendations when Cart changes
  useEffect(() => {
    const fetchRecs = async () => {
      if (cart.length === 0) {
        setRecommendations([])
        return
      }
      
      setLoadingRecs(true)
      try {
        const itemIds = cart.map(item => item.id)
        const res = await axios.post(`${API_URL}/recommend`, {
          basket_item_ids: itemIds
        })
        
        // Mock integration if API is returning empty for now
        if (res.data.recommendations && res.data.recommendations.length > 0) {
            setRecommendations(res.data.recommendations)
        } else {
            // Fake Mock Data cho đến khi kết nối với Deep Learning module
            setTimeout(() => {
                setRecommendations([
                  {id: "mock1", title: "Sản phẩm bổ trợ A (Mock Real-time)", price: "$25.00", image: "https://placehold.co/150x150/e0e7ff/4338ca?text=Product+A", confidence: 95.2},
                  {id: "mock2", title: "Sản phẩm bổ trợ B (Mock Real-time)", price: "$15.00", image: "https://placehold.co/150x150/e0e7ff/4338ca?text=Product+B", confidence: 88.7},
                  {id: "mock3", title: "Sản phẩm bổ trợ C (Mock Real-time)", price: "$45.00", image: "https://placehold.co/150x150/e0e7ff/4338ca?text=Product+C", confidence: 82.1}
                ])
                setLoadingRecs(false)
            }, 800)
            return;
        }

      } catch (err) {
        console.error("Error loading recommendations:", err)
      } finally {
        setLoadingRecs(false)
      }
    }
    
    fetchRecs()
  }, [cart])

  const isInCart = (id) => cart.some(item => item.id === id)

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <header className="bg-slate-900 text-white p-4 sticky top-0 z-50 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Sparkles className="text-yellow-400" />
            <h1 className="text-xl font-bold font-sans tracking-tight">AI Shop Demo</h1>
          </div>
          <div className="flex items-center gap-3 bg-slate-800 px-4 py-2 rounded-full cursor-pointer hover:bg-slate-700 transition"
               onClick={() => document.getElementById('cart-section').scrollIntoView({behavior: 'smooth'})}>
            <ShoppingCart size={20} />
            <span className="font-semibold">{cart.length} món</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 mt-8 space-y-12">
        
        {/* Cold Start Section */}
        <section>
          <div className="mb-6">
            <h2 className="text-2xl font-extrabold text-gray-800 flex items-center gap-2">
              <Sparkles className="text-blue-500"/>
              Dành riêng cho bạn (Cold-Start)
            </h2>
            <p className="text-gray-500">20 sản phẩm nổi bật nhất được trích xuất từ Big Data Spark.</p>
          </div>
          
          {loadingInitial ? (
            <div className="flex justify-center items-center h-40">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-slate-900"></div>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {seedProducts.map(product => (
                <div key={product.id} className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition">
                  <div className="h-40 w-full mb-3 overflow-hidden rounded-lg bg-gray-100 flex items-center justify-center">
                    <img src={product.image} alt="product" className="object-contain max-h-full mix-blend-multiply" />
                  </div>
                  <h3 className="font-medium text-sm text-gray-800 line-clamp-2 min-h-[40px]">{product.title}</h3>
                  <div className="mt-3 flex items-center justify-between">
                    <span className="font-bold text-red-600">{product.price}</span>
                    <button 
                      onClick={() => isInCart(product.id) ? removeFromCart(product.id) : addToCart(product)}
                      className={`p-2 rounded-full transition-colors ${isInCart(product.id) ? 'bg-red-50 text-red-600 hover:bg-red-100' : 'bg-slate-900 text-white hover:bg-slate-800'}`}
                    >
                      {isInCart(product.id) ? <Trash2 size={16} /> : <Plus size={16} />}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Cart & Realtime Deep Learning Recommendations */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8" id="cart-section">
          
          {/* Cart View */}
          <section className="col-span-1 bg-white p-6 rounded-2xl shadow-sm border border-gray-100 h-fit">
            <h2 className="text-xl font-bold flex items-center gap-2 mb-4 border-b pb-4">
              <ShoppingCart className="text-slate-800" />
              Giỏ hàng đang chọn
            </h2>
            
            {cart.length === 0 ? (
              <div className="text-center py-10 text-gray-400">
                <Box size={48} className="mx-auto mb-3 opacity-50" />
                <p>Giỏ hàng trống.</p>
                <p className="text-sm">Hãy chọn vài sản phẩm ở trên!</p>
              </div>
            ) : (
              <div className="space-y-4">
                {cart.map(item => (
                  <div key={item.id} className="flex gap-3 items-center bg-gray-50 p-3 rounded-lg">
                    <img src={item.image} className="w-12 h-12 object-cover rounded bg-white mix-blend-multiply" alt=""/>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-sm truncate">{item.title}</p>
                      <p className="font-semibold text-red-600 text-sm">{item.price}</p>
                    </div>
                    <button onClick={() => removeFromCart(item.id)} className="text-gray-400 hover:text-red-500">
                      <Trash2 size={18} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* Deep Learning Recs View */}
          <section className="col-span-1 lg:col-span-2 bg-gradient-to-br from-indigo-50 to-blue-50 p-6 rounded-2xl shadow-sm border border-indigo-100">
            <h2 className="text-xl font-bold text-indigo-900 flex items-center gap-2 mb-2">
              <Sparkles className="text-indigo-600" />
              Gợi ý Mua kèm (Real-time Deep Learning)
            </h2>
            <p className="text-sm text-indigo-700 mb-6">Mô hình AI tự động phân tích các món đồ trong giỏ để dự đoán nhu cầu tiếp theo của bạn.</p>
            
            {cart.length === 0 ? (
              <div className="h-64 flex items-center justify-center text-indigo-400 border-2 border-dashed border-indigo-200 rounded-xl">
                 Đang chờ tín hiệu từ giỏ hàng...
              </div>
            ) : loadingRecs ? (
              <div className="h-64 flex flex-col gap-3 items-center justify-center text-indigo-600">
                 <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600"></div>
                 <p className="animate-pulse font-medium text-sm">Đang suy luận qua Mạng Neural Vec2Seq...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {recommendations.map(rec => (
                  <div key={rec.id} className="bg-white p-4 rounded-xl shadow-sm hover:shadow-md transition relative overflow-hidden group">
                    <div className="absolute top-0 right-0 bg-indigo-600 text-white text-[10px] font-bold px-2 py-1 rounded-bl-lg z-10">
                      Độ tự tin: {rec.confidence}%
                    </div>
                    <div className="h-32 w-full mb-3 overflow-hidden rounded-lg bg-gray-50 flex items-center justify-center">
                      <img src={rec.image} alt="recommendation" className="object-contain max-h-full mix-blend-multiply group-hover:scale-110 transition duration-300" />
                    </div>
                    <h3 className="font-medium text-sm text-gray-800 line-clamp-2 min-h-[40px]">{rec.title}</h3>
                    <div className="mt-2 text-red-600 font-bold text-lg">{rec.price}</div>
                  </div>
                ))}
              </div>
            )}
            
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
