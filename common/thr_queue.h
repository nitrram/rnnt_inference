#pragma once

#include <queue>
#include <mutex>

#include "common/buf_type.h"

namespace spr::inference {

	
	using buf_size_t = std::vector<buf_t>;


	class thr_queue {
	public:

		virtual ~thr_queue() = default;

		buf_size_t pop();

		void push(buf_size_t &&);

		bool empty();

		size_t size();
		
	private:
		std::queue<buf_size_t> m_queue;
		std::mutex m_mutex;
	};
}
