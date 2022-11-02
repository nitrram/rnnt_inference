#include "thr_queue.h"


namespace spr::inference {

	buf_size_t thr_queue::pop() {

		std::lock_guard<std::mutex> lck(m_mutex);

		auto res = m_queue.front();

		m_queue.pop();
		
		return res;
	}

	void thr_queue::push(buf_size_t &&el) {

		std::lock_guard<std::mutex> lck(m_mutex);

		m_queue.push(el);
	}

	bool thr_queue::empty() {
		std::lock_guard<std::mutex> lck(m_mutex);

		return m_queue.empty();
	}

	size_t thr_queue::size() {
		std::lock_guard<std::mutex> lck(m_mutex);

		return m_queue.size();
	}
}
